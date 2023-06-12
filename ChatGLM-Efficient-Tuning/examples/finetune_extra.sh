#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /kaggle/working/model/src/finetune.py \
    --do_train \
    --model_name_or_path THUDM/chatglm-6b-int4 \
    --dataset datasetExtraExtra \
    --dataset_dir /kaggle/working/model/data \
    --finetuning_type p_tuning \
    --output_dir /kaggle/working/model/checkpoint4 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --quantization_bit 4 \
    --checkpoint_dir /kaggle/working/model/checkpoint3/checkpoint-5000 \
    --max_source_length 2048 \
    --max_target_length 1024
    # --fp16 \
