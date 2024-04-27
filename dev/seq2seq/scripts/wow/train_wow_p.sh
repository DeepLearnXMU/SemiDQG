#!/usr/bin/env bash

# script to run prior & posterior model

DATA_DIR='../../saved_data/data_woi'
WOW_DIR='../../saved_data/data_wow'
OUTPUT_DIR='../../ckpt'

CUDA_VISIBLE_DEVICES=5 python run_model.py \
    --model_name_or_path t5-base \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$WOW_DIR/valid.json" \
    --test_file "$WOW_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/t5-base-wow-p" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --logging_steps 100 \
    --num_train_epochs 6 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --metric_for_best_model rouge1 \
    --load_best_model_at_end \
    --weight_decay 0.01 \
    --text_column="dialogue" \
    --summary_column="query" \
    --do_lower_case \
    --remove_unused_columns false \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir
