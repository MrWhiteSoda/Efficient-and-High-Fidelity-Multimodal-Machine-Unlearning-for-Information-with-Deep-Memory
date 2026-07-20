#!/bin/bash

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
# MASTER_PORT=6001
MASTER_PORT=6001

MODEL="../output/minicpmv_sensitive_foi_fci"
MODEL="../output/minicpmv_sensitive_foi_fci"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.

DATA="../data/vqa/FOI_FCI/FOI_FCI_100.json"
EVAL_DATA="../data/vqa/FOI_FCI/FOI_FCI_100.json"
LLM_TYPE="minicpm" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",if use openbmb/MiniCPM-V-2_6, please set LLM_TYPE="qwen2"
MODEL_MAX_Length=1536 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

CUDA_VISIBLE_DEVICES=2,3 torchrun $DISTRIBUTED_ARGS finetune_sensitive_kul.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision false \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 125 \
    --eval_steps 25 \
    --output_dir ../output/minicpmv_sensitive_foi_fci_kul \
    --logging_dir ../output/minicpmv_sensitive_foi_fci_kul \
    --logging_strategy "steps" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 125 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ../config/ds_config_zero2.json \
    --report_to "tensorboard"