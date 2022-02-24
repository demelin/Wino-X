#!/bin/sh

# Variables
src_lang=en
tgt_lang=$1
data_dir=$2
model_dir=$3
init_ckpt=$4


mkdir $model_dir
mkdir $model_dir/checkpoints
mkdir $model_dir/tensorboard_logs


# Train / finetune model
# For hyper-parameter choices, see https://github.com/pytorch/fairseq/issues/346
# NOTE: PARAMETERS ASSUME TRAINIG ON TWO GPUS!
fairseq-train $data_dir \
    --arch transformer \
    --source-lang $src_lang \
    --target-lang $tgt_lang \
    --max-tokens 4096 \
    --update-freq 3 \
    --share-all-embeddings \
    --dropout 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)'\
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --hinge-lambda 100 \
    --label-smoothing 0.1 \
    --weight-decay 0.0 \
    --clip-norm 0.0 \
    --validate-interval-updates 10000 \
    --log-format json \
    --log-interval 1000 \
    --save-interval-updates 10000 \
    --save-dir $model_dir/checkpoints \
    --tensorboard-logdir $model_dir/tensorboard_logs \
    --max-update 1000000 \
    --patience 3
#    --criterion label_smoothed_cross_entropy_with_pronoun_hinge_loss \
#    --finetune-from-model $init_ckpt
