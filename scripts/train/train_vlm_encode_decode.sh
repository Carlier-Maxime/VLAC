export BATCH_SIZE=8
export ACC_STEP=1
export bs=$((BATCH_SIZE / ACC_STEP))
export TOKENIZERS_PARALLELISM="true"

torchrun --nproc_per_node=1 --master_port=25001 \
    -m vlac.train.train \
    --version v1 \
    --trainer_type EncodeDecodeTrainer \
    --model_name_or_path ./vlac_base \
    --data_mixture embeds \
    --chunk_sampler True \
    --bf16 True \
    --output_dir /media/hdd/maxime_carlier/checkpoints/vlac/train-vlm-encode-decode \
    --num_train_epochs 16 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $ACC_STEP \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 32 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --split_batches True
