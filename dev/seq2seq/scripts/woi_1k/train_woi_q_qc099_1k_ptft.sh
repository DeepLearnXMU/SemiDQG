#!/usr/bin/env bash

# script to run prior & posterior model

DATA_DIR='../../saved_data/data_woi'
OUTPUT_DIR='../../ckpt'
CKPT_NAME='t5-base-woi-q-qc099-1k-ptft'

CUDA_VISIBLE_DEVICES=6 python run_model.py \
    --model_name_or_path "$OUTPUT_DIR/t5-base-woi-q-qc099-1k-pt-step5k/checkpoint-5000" \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train_1k.json" \
    --validation_file "$DATA_DIR/valid.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/$CKPT_NAME" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --logging_steps 100 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --metric_for_best_model rouge1 \
    --load_best_model_at_end \
    --weight_decay 0.01 \
    --text_column="dialogue" \
    --summary_column="query" \
    --do_lower_case \
    --use_posterior \
    --remove_unused_columns false \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir

python eval_txt.py woi $OUTPUT_DIR/$CKPT_NAME/generated_predictions.txt test