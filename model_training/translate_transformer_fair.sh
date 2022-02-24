#!/bin/sh
export LC_CTYPE=en_US.UTF-8

# Corpus-specific variables
tgt_lang=$1
input_file=$2
output_file=$3
checkpoint_path=$4
codes_path=$5
vocab_dir=$6


# Translate
{
fairseq-interactive \
    --path $checkpoint_path $vocab_dir \
    --input $input_file \
    --tokenizer moses \
    --remove-bpe \
    --bpe fastbpe \
    --bpe-codes $codes_path \
    --batch-size 8 \
    --buffer-size 8 \
    --beam 5 \
    --source-lang en \
    --target-lang $tgt_lang
} > $output_file.with_scores


# Clean
python3 ./fairseq_output_cleaner.py $output_file.with_scores $output_file $tgt_lang

