 import torch
import torch.nn as nn
import deepspeed
# from apex import amp
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *
from transformers.integrations import is_deepspeed_zero3_enabled


class CPMTrainer(Trainer):
    def __init__(self, mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def _get_weight_mask(self, model, dataloader, optimizer, device):

        gradients = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param.data).cpu()

        model.eval()

        for batch in dataloader:
            unlearning_batch = batch['unlearning']

            loss = self._compute_batch_loss(model, unlearning_batch)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            optimizer.zero_grad()
            if self.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient = param.grad.data.cpu()
                        gradients[name] += gradient

            del unlearning_batch
            torch.cuda.empty_cache()

        with torch.no_grad():
            for name in gradients:
                # print(name)
                # print(gradients[name])
                gradients[name] = torch.abs_(gradients[name])

            threshold_list = [0.5]
            for i in threshold_list:
                # print(i)
                sorted_dict_positions = {}
                hard_dict = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat(
                    [tensor.flatten() for tensor in gradients.values()]
                )

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index: start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements

        return hard_dict

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

        mismatch_batch = inputs['mismatch']
        match_batch = inputs['match']

        alpha = 1e-3

        unlearning_loss = self._compute_batch_loss(self.model, mismatch_batch)
        match_loss = self._compute_batch_loss(self.model, match_batch)
        loss = unlearning_loss + alpha * match_loss

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
        if self.state.epoch == 0.0:
            self.mask = self._get_weight_mask(self.model, self.get_train_dataloader(), self.optimizer,
                                              self.model.device)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if self.mask:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad *= self.mask[name].to(param.grad.device)

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