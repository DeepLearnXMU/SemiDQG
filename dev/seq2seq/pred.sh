#!/usr/bin/env bash
LANG='en'
DATASET='wow'
DATA_DIR='../../saved_data/data_'$DATASET
MODEL_PATH='../../ckpt/t5-base-wow-p'

CUDA_VISIBLE_DEVICES=6 python run_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_predict \
    --test_file "$DATA_DIR/test.json" \
    --predict_output_file "generated_predictions.txt" \
    --source_prefix "" \
    --output_dir ${MODEL_PATH} \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --text_column="dialogue" \
    --summary_column="query" \
    --do_lower_case \
    --remove_unused_columns false \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir \
    --lang ${LANG} # --use_posterior

python eval_txt.py $DATASET $MODEL_PATH/generated_predictions.txt test