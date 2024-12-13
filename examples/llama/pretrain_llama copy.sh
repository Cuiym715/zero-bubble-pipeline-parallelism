#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

TS=`date +%Y_%m_%d_%H_%M_%S`

DATA_PATH="/workspace/data/zb_sample_dataset/dataset/c4_text_document"
# ~/dataset/enwiki-100m_text_document

export WORLD_SIZE=$NUM_GPUS

if [ $GQA == "1" ]; then
    GQA_ARGS="--group-query-attention --num-query-groups $NUM_QUERY_GROUPS"
fi

if [ $CKPT == "full" ]; then
    CKPT_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
elif [ $CKPT == "no" ]; then
    CKPT_ARGS=
fi

# TODO: 没下载transformer engine，而且也没有tp
# if [ $CP == "1" ]; then
#     TP_OVERLAP_ARGS="--tp-comm-overlap"
# fi

profile_ranks="0"
for ((i = 1; i < $WORLD_SIZE; i++)); do
    profile_ranks="$profile_ranks $i"
done

# PP schedule
PP_ARGS=
if [ $PP_SCHEDULE == "gpipe" ]; then
    echo "gpipe"
    sleep 3
    PP_ARGS="--use-gpipe"
elif [ $PP_SCHEDULE == "pipeline" ]; then
    PP_ARGS="--pipeline-schedule"
elif [ $PP_SCHEDULE == "hanayo" ]; then
    PP_ARGS="--num-waves-per-pipeline $WAVE_SIZE"
elif [ $PP_SCHEDULE == "v-zb" ]; then
    ENABLE_ZERO_BUBBLE=1
    PP_ARGS="--zero-bubble-v-schedule"
elif [ $PP_SCHEDULE == "v-half" ]; then
    echo "v-half"
    sleep 3
    ENABLE_ZERO_BUBBLE=1
    PP_ARGS="--zero-bubble-v-schedule \
    --zero-bubble-v-schedule-mem-setup half"
elif [ $PP_SCHEDULE == "chimera" ]; then
    echo "chimera"
    sleep 3
    PP_ARGS="--enable-chimera"
else
    echo "Invalid PP_SCHEDULE"
    sleep 10
fi
# zero bubble settings
if [ ! -z "${ENABLE_ZERO_BUBBLE:-}" ]; then
    if [ -z "${ZERO_BUBBLE_TIMER_START:-}" ]; then
        ZERO_BUBBLE_TIMER_START=100
        ZERO_BUBBLE_TIMER_END=110
        ZERO_BUBBLE_TIMER_START=10
        ZERO_BUBBLE_TIMER_END=20
    fi
    PP_ARGS="$PP_ARGS --enable-zero-bubble \
    --zero-bubble-pipeline-timers-start-iter $ZERO_BUBBLE_TIMER_START \
    --zero-bubble-pipeline-timers-end-iter $ZERO_BUBBLE_TIMER_END \
    --zero-bubble-max-pending-backward $ZERO_BUBBLE_MEM_LIMIT"
    PP_ARGS="$PP_ARGS --fp16 --enable-optimizer-post-validation"
    # if [ -z "${FP32:-}" ]; then
    #     PP_ARGS="$PP_ARGS --fp16 --enable-optimizer-post-validation"
    # fi
else
    PP_ARGS="$PP_ARGS --fp16"
fi

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    ...
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    ...
"

PP_ARGS="$PP_ARGS --num-waves-per-pipeline $WAVE_SIZE"

mkdir -p logs
set -x
mpirun --allow-run-as-root \
        ${CLUSTER_MPI_ARGS:-} \
        --mca btl self,tcp \
        --mca pml ob1 \
        --np $NUM_GPUS \
        --bind-to none --map-by slot \
        -x MPI_THREAD_SINGLE=1 \
        -x NCCL_DEBUG=WARN \
        -x PYTHONPATH=../../ \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NVTE_FWD_LAYERNORM_SM_MARGIN=8 \
        -x NVTE_BWD_LAYERNORM_SM_MARGIN=8 \
        -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        -x PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 \
        -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=6005 \
    python3 ../../pretrain_gpt.py \
    $PP_ARGS \
    $CKPT_ARGS \
    ${TP_OVERLAP_ARGS:-} \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TIMING_ARGS \
    2>&1 | tee logs/llama_$TS.txt