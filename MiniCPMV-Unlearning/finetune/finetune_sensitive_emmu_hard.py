import glob
import json
import logging
import os
import time
import datetime
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
from transformers import AutoModel, AutoTokenizer

from dataset import SupervisedDataset, SupervisedDatasetHard, DisturbanceDataset, EMMUDatasetHard, emmu_data_collator_hard1
from trainer_sensitive_emmu_hard import CPMTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    original_model_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")
    unlearned_model_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:

    unlearning_path: str = field(
        default=None, metadata={"help": "Path to the unlearning data."}
    )
    disturbance_path: str = field(
        default=None, metadata={"help": "Path to the disturbance data."}
    )
    mismatch_path: str = field(
        default=None, metadata={"help": "Path to the mismatch data."}
    )
    match_path: str = field(
        default=None, metadata={"help": "Path to the match data."}
    )

    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    target_layers: Optional[str] = field(default="all")
    param_path: Optional[str] = field(default=None)
    param_scores_path: Optional[str] = field(default=None)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir, )


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        transform,
        data_collator=None,
        llm_type="minicpm",
        slice_config=None,
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
) -> Dict:
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    unlearning_json = json.load(open(data_args.unlearning_path, "r"))
    unlearning_dataset = dataset_cls(
        unlearning_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    disturbance_json = json.load(open(data_args.disturbance_path, "r"))
    disturbance_dataset = DisturbanceDataset(disturbance_json)

    mismatch_json = json.load(open(data_args.mismatch_path, "r"))
    mismatch_dataset = dataset_cls(
        mismatch_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    match_json = json.load(open(data_args.match_path, "r"))[:38]
    match_dataset = dataset_cls(
        match_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    train_dataset = EMMUDatasetHard(unlearning_dataset, disturbance_dataset, mismatch_dataset, match_dataset)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(data_collator, max_length=max_length),
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0

param_indices = None

def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    original_model = AutoModel.from_pretrained(
        model_args.original_model_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )
    unlearned_model = AutoModel.from_pretrained(
        model_args.unlearned_model_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.unlearned_model_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        unlearned_model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        unlearned_model.llm.requires_grad_(False)

    for param in original_model.parameters():
        param.requires_grad_(False)
    for param in unlearned_model.parameters():
        param.requires_grad_(False)

    if training_args.target_layers == "all":
        target_layers = list(range(unlearned_model.config.num_hidden_layers))
    else:
        target_layers = [int(x) for x in training_args.target_layers.split(",")]

    for name, param in unlearned_model.llm.named_parameters():
        for layer_idx in target_layers:
            if f"layers.{layer_idx}.mlp.up_proj" in name:
                param.requires_grad_(True)
                break

    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")

        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in unlearned_model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens', 'resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(unlearned_model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()

            unlearned_model.get_input_embeddings = MethodType(get_input_embeddings, unlearned_model)
        if lora_args.q_lora:
            unlearned_model = prepare_model_for_kbit_training(
                unlearned_model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        unlearned_model = get_peft_model(unlearned_model, lora_config)
        if training_args.gradient_checkpointing:
            unlearned_model.enable_input_require_grads()

    rank0_print(get_parameter_number(unlearned_model))
    llm_type = training_args.llm_type
    rank0_print(f'llm_type={llm_type}')

    # Load data
    if hasattr(unlearned_model.config, "slice_config"):
        unlearned_model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = unlearned_model.config.slice_config.to_dict()
    else:
        unlearned_model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = unlearned_model.config.to_dict()

    if hasattr(unlearned_model.config, "batch_vision_input"):
        batch_vision = unlearned_model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=transform_func,
        data_collator=emmu_data_collator_hard1,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=unlearned_model.config.patch_size,
        query_nums=unlearned_model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )

    if training_args.param_path:
        crucial_parameters = torch.load(training_args.param_path)
        params_pool = {}
        for idx, parameters in enumerate(crucial_parameters):
            params_pool[f'model.layers.{target_layers[idx]}.mlp.up_proj.weight'] = parameters

    else:
        params_pool = None
    if training_args.param_scores_path:
        param_scores = torch.load(training_args.param_scores_path)
    else:
        param_scores = None

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = CPMTrainer(
        original_model=original_model,
        params_pool=params_pool,
        param_scores=param_scores,
        target_layers=target_layers,
        slice_config=slice_config,
        model=unlearned_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    rank0_print('Training time {}'.format(str(datetime.timedelta(seconds=int(training_time)))))


    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()