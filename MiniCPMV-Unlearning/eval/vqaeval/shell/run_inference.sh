export CUDA_VISIBLE_DEVICES="2,3"
# --eval_docVQATest \
  #    --docVQATest_image_dir ./downloads/DocVQA/spdocvqa_images \
  #    --docVQATest_ann_path ./downloads/DocVQA/test_v1.0.json \
#       --eval_textVQA \
#       --textVQA_image_dir ./downloads/TextVQA/train_images \
#       --textVQA_ann_path ./downloads/TextVQA/TextVQA_0.5.1_val.json \
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-2} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12395} \
    ./eval.py \
    --model_name minicpmv \
    --model_path ../../output/minicpmv_sensitive_foi_emmu \
    --generate_method interleave \
    --eval_docVQA \
    --docVQA_image_dir ./downloads/DocVQA/spdocvqa_images \
    --docVQA_ann_path ./downloads/DocVQA/val_v1.0_withQT.json \
    --answer_path ./outputs/minicpmv_sensitive_foi_emmu \
    --batchsize 1