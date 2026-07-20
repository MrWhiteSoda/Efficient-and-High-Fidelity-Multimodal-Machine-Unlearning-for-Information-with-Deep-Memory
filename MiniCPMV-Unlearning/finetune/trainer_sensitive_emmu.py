import time
import datetime
import torch
import torch.nn as nn
import deepspeed
import json
# from apex import amp
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *
from transformers.integrations import is_deepspeed_zero3_enabled
import shap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class CPMTrainer(Trainer):
    def __init__(self, original_model=None, params_pool=None, target_layers=None, slice_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.accelerator.device
        original_model = original_model.to(device).to(torch.bfloat16)
        self.original_model = original_model
        self.slice_config = slice_config
        self.params_pool = params_pool
        self.target_layers = target_layers
        self.batch_params_indices = {}

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _custom_generate(
        self,
        model,
        data_list=None,
        img_list=None,
        tokenizer=None,
        max_inp_length=None,
        vision_hidden_states=None,
        **kwargs
    ):

        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = model._process_list(tokenizer, data_list, max_inp_length)
        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(model.transform(img).to(model.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                _,
            ) = model.get_vllm_embedding(model_inputs)

            result = model.llm.generate(inputs_embeds=model_inputs["inputs_embeds"], pad_token_id=0,
                                        eos_token_id=tokenizer.eos_token_id, **kwargs)
            # del model_inputs
        torch.cuda.empty_cache()

        return result

    def _custom_chat(
        self,
        model,
        images,
        msgs,
        tokenizer,
        vision_hidden_states=None,
        max_new_tokens=256,
        sampling=True,
        max_inp_length=2048,
        **kwargs
    ):

        if isinstance(msgs, str):
            msgs = json.loads(msgs)
            # msgs to prompt
        prompts = []
        input_images = []
        for i, msg in enumerate(msgs):
            prompt = ""
            role = msg["role"]
            content = msg["content"]
            assert role == "user", "The role of first msg should be user"
            # print(self.slice_config)
            if self.slice_config['slice_mode']:
                imgs, final_placeholder = model.get_slice_image_placeholder(
                    images[i], tokenizer
                )
                content = final_placeholder + "\n" + content
            else:
                imgs = [images[i]]
                content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * model.config.query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                )
            input_images.append(imgs)
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
            prompt += "<AI>"
            prompts.append(prompt)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res = self._custom_generate(
                model=model,
                data_list=prompts,
                max_inp_length=max_inp_length,
                img_list=input_images,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                **generation_config
            )
            # del prompts, input_images

        torch.cuda.empty_cache()

        return res


    def _compute_disturbance_outputs(self, model, images, msgs, target_layers, tokenizer, device):

        ffn_outputs = []

        def capture_ffn_output(module, input, output):
            if len(ffn_outputs) < len(target_layers):
                ffn_outputs.append(output[:, -1, :].cpu())

        def register_ffn_hook(model, target_layers):
            hooks = []
            for idx, layer in enumerate(model.llm.model.layers):
                if idx in target_layers:
                    hook = layer.mlp.up_proj.register_forward_hook(capture_ffn_output)
                    hooks.append(hook)
            return hooks

        hooks = register_ffn_hook(model, target_layers)

        _ = self._custom_chat(
            model=model,
            images=images,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            max_new_tokens=1,
        )

        for hook in hooks:
            hook.remove()

        torch.cuda.empty_cache()

        return ffn_outputs


    def _parameters_input_solver(self, unlearning_outputs, disturbance_outputs, target_layers):

        layer_indices = []

        unlearning_outputs_tensor = torch.stack(unlearning_outputs, dim=0)
        disturbance_outputs_tensor = torch.stack(disturbance_outputs, dim=0)

        scores = torch.mean(torch.abs(unlearning_outputs_tensor - disturbance_outputs_tensor), dim=1)
        for idx, layer_idx in enumerate(target_layers):
            if layer_idx < 25:
                layer_indices.append(torch.where(scores[idx] >= 0.2)[0].tolist())
            else:
                layer_indices.append(torch.topk(scores[idx], 2500).indices.tolist())

        del unlearning_outputs_tensor, disturbance_outputs_tensor, scores
        torch.cuda.empty_cache()

        return layer_indices


    def _compute_ffn_outputs_only_input(self, model, disturbance_batch, target_layers, tokenizer, device):
        images = disturbance_batch['images']
        disturbance_images = disturbance_batch['disturbance_images']
        msgs = disturbance_batch['msgs']
        unlearning_first_token_outputs = self._compute_disturbance_outputs(
            model, images, msgs, target_layers, tokenizer, device
        )
        torch.cuda.empty_cache()
        disturbance_first_token_outputs = self._compute_disturbance_outputs(
            model, disturbance_images, msgs, target_layers, tokenizer, device
        )
        torch.cuda.empty_cache()
        return unlearning_first_token_outputs, disturbance_first_token_outputs


    def _get_parameter_indices_only_input(self, model, data_batch, target_layers, tokenizer, device):
        model.eval()
        unlearning_first_token_outputs, disturbance_outputs = self._compute_ffn_outputs_only_input(
            model, data_batch, target_layers, tokenizer, device)

        parameter_indices = self._parameters_input_solver(unlearning_first_token_outputs, disturbance_outputs,
                                                      target_layers)
        torch.cuda.empty_cache()

        return parameter_indices



    def _compute_batch_loss(self, model, batch, return_outputs=False):
        if "labels" in batch:
            labels = batch.pop("labels")
        else:
            labels = None

        if not self.args.use_lora:
            outputs = model(data=batch, use_cache=False)
        else:
            with model._enable_peft_forward_hooks(**batch):
                outputs = model.base_model(data=batch, use_cache=False)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1, model.config.vocab_size).contiguous()
            labels = labels.view(-1).long().contiguous()
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(batch.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):

        unlearning_batch = inputs['unlearning']
        mismatch_batch = inputs['mismatch']
        match_batch = inputs['match']
        gamma1 = 0.1
        gamma2 = 1.0
        gamma3 = 0.5


        unlearning_loss = -1.0 * self._compute_batch_loss(self.model, unlearning_batch)


        if "labels" in match_batch:
            _ = match_batch.pop("labels")
        if not self.args.use_lora:
            unlearning_model_outputs = self.model(data=match_batch, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**match_batch):
                unlearning_model_outputs = self.model.base_model(data=match_batch, use_cache=False)
        with torch.no_grad():
            original_model_outputs = self.original_model(data=match_batch, use_cache=False)
        prob_p = torch.nn.functional.softmax(original_model_outputs.logits, dim=-1)
        prob_q = torch.nn.functional.softmax(unlearning_model_outputs.logits, dim=-1)
        remain_loss = (prob_p * (torch.log(prob_p + 1e-12) - torch.log(prob_q + 1e-12))).sum(dim=-1).mean()

        mismatch_loss = self._compute_batch_loss(self.model, mismatch_batch)
        loss = gamma1 * unlearning_loss + gamma2 * remain_loss + gamma3 * mismatch_loss
        torch.cuda.empty_cache()
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                         for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self._compute_batch_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)


        parameter_indices = self._get_parameter_indices_only_input(model, inputs['disturbance'], self.target_layers, self.tokenizer, self.args.device)
        params_in_pool = []
        for layer_idx, indices in enumerate(parameter_indices):
            weight_name = f'model.layers.{self.target_layers[layer_idx]}.mlp.up_proj.weight'
            if weight_name in self.params_pool.keys():
                param_pool = self.params_pool[weight_name]
                indices_tensor = torch.tensor(indices, device=param_pool.device, dtype=torch.long)
                mask = param_pool[indices_tensor, :].any(dim=1)
                params_in_pool.append(indices_tensor[mask].tolist())

        epoch_step = self.state.global_step % self.state.save_steps
        if self.state.global_step < self.state.save_steps:
            self.batch_params_indices[epoch_step] = params_in_pool
        else:
            last_params_in_pool = self.batch_params_indices[epoch_step]
            need_update_param = [list(set(current).difference(set(last))) for last, current in
                                 zip(last_params_in_pool, params_in_pool)]
            for layer_idx, indices in enumerate(need_update_param):
                weight_name = f'model.layers.{self.target_layers[layer_idx]}.mlp.up_proj.weight'
                if weight_name in self.params_pool:
                    self.params_pool[weight_name][indices, :] = 0
            self.batch_params_indices[epoch_step] = params_in_pool

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if self.params_pool:
            for name, param in self.model.llm.named_parameters():
                if name in self.params_pool.keys():
                    param.grad *= self.params_pool[name].to(self.model.device)


        return loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:

            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))