export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export BATCH_SIZE=8
export ACC_STEP=1
export bs=$((BATCH_SIZE / ACC_STEP))
export TOKENIZERS_PARALLELISM="true"

torchrun --nproc_per_node=2 --master_port=25001 \
    -m vlac.train.train \
    --version v1 \
    --data_mixture sharegpt4v_pretrain+mmc4core+openvid_generation \
    --chunk_sampler True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_vi_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vlac-train-vision_tower \
    --num_train_epochs 8 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $ACC_STEP \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 4 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --accelerator_config "{\"split_batches\": true}"
