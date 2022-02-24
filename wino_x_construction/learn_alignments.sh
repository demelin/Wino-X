#!/bin/sh

DATA_FILE=$1
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE=$2

CUDA_VISIBLE_DEVICES=0 python /xxx/awesome-align/run_align.py \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32
