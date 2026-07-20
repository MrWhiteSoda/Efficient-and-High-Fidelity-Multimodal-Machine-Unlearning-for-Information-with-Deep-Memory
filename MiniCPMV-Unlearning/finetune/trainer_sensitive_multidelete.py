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
    def __init__(self, original_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.accelerator.device
        original_model = original_model.to(device).to(torch.bfloat16)
        self.original_model = original_model

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


    def _compute_logits(self, model, batch):
        if "labels" in batch:
            labels = batch.pop("labels")
        else:
            labels = None

        if not self.args.use_lora:
            outputs = model(data=batch, use_cache=False)
        else:
            with model._enable_peft_forward_hooks(**batch):
                outputs = model.base_model(data=batch, use_cache=False)

        logits = outputs.logits.view(-1, model.config.vocab_size).contiguous()

        return logits


    def compute_loss(self, model, inputs, return_outputs=False):

        unlearning_batch = inputs['unlearning']
        unlearning_text_batch = inputs['unlearning_text']
        mismatch_batch = inputs['mismatch']
        match_batch = inputs['match']

        md_weight = 1.0
        mkr_weight = 1.0
        ukr_weight = 1.0

        with torch.no_grad():
            original_model_unlearning_logits = self._compute_logits(self.original_model, unlearning_batch)
        unlearned_model_unlearning_logits = self._compute_logits(self.model, mismatch_batch)

        md_loss = torch.nn.functional.mse_loss(original_model_unlearning_logits, unlearned_model_unlearning_logits)
        with torch.no_grad():
            original_model_match_logits = self._compute_logits(self.original_model, match_batch)
        unlearned_model_match_logits = self._compute_logits(self.model, match_batch)
        mkr_loss = torch.nn.functional.mse_loss(original_model_match_logits, unlearned_model_match_logits)
        with torch.no_grad():
            if "labels" in unlearning_batch:
                labels = unlearning_batch.pop("labels")
            _, original_model_unlearning_vision_logits = self.original_model.get_vllm_embedding(unlearning_batch)
            original_model_unlearning_vision_logits = torch.stack(original_model_unlearning_vision_logits)
            original_model_unlearning_vision_logits = original_model_unlearning_vision_logits.view(-1, original_model_unlearning_vision_logits.shape[-1])
            position_ids = unlearning_text_batch["position_ids"]
            if position_ids.dtype != torch.int64:
                position_ids = position_ids.long()
            original_model_llm_logits = self.original_model.llm(
                input_ids=unlearning_text_batch["input_ids"],
                attention_mask=unlearning_text_batch["attention_mask"],
                position_ids=position_ids,
                use_cache=False
            ).logits
            original_model_llm_logits = original_model_llm_logits.view(-1, self.original_model.config.vocab_size).contiguous()
        _, unlearned_model_unlearning_vision_logits = self.model.get_vllm_embedding(unlearning_batch)
        unlearned_model_unlearning_vision_logits = torch.stack(unlearned_model_unlearning_vision_logits)
        unlearned_model_unlearning_vision_logits = unlearned_model_unlearning_vision_logits.view(-1, unlearned_model_unlearning_vision_logits.shape[-1])
        unlearned_model_llm_logits = self.model.llm(
            input_ids=unlearning_text_batch["input_ids"],
            attention_mask=unlearning_text_batch["attention_mask"],
            position_ids=position_ids,
            use_cache=False
        ).logits
        unlearned_model_llm_logits = unlearned_model_llm_logits.view(-1, self.model.config.vocab_size).contiguous()
        ukr_loss = torch.nn.functional.mse_loss(original_model_unlearning_vision_logits, unlearned_model_unlearning_vision_logits) + \
                     torch.nn.functional.mse_loss(original_model_llm_logits, unlearned_model_llm_logits)
        loss = md_weight * md_loss + mkr_weight * mkr_loss + ukr_weight * ukr_loss

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

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

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