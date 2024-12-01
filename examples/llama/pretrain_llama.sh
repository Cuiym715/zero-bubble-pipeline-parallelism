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
    PP_ARGS="--use-gpipe"
elif [ $PP_SCHEDULE == "pipeline" ]; then
    PP_ARGS="--pipeline-schedule"
elif [ $PP_SCHEDULE == "v-zb" ]; then
    ENABLE_ZERO_BUBBLE=1
    PP_ARGS="--zero-bubble-v-schedule"
elif [ $PP_SCHEDULE == "v-half" ]; then
    echo "v-half"
    sleep 3
    ENABLE_ZERO_BUBBLE=1
    PP_ARGS="--zero-bubble-v-schedule \
    --zero-bubble-v-schedule-mem-setup half"
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
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    ${GQA_ARGS:-} \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 1.5e-4 \
    --train-iters $TRAIN_ITERS \
    --exit-interval 1000 \
    --lr-decay-iters 500000 \
    --lr-decay-style cosine \
    --min-lr 1.5e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --clip-grad 8.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --profile \
    --profile-step-start $PROFILE_STEP_START \
    --profile-step-end $PROFILE_STEP_END \
    --profile-ranks $profile_ranks \
    --allow-padding-num-layers \
    --no-barrier-with-level-1-timing \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /workspace/data/zb_sample_dataset/tokenizers/tokenizer.model \
    --vocab-size 32004 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

TIMING_ARGS="
    --timers-save $TIMER_SAVE_PATH \
    --timing-log-level $TIMING_LOG_LEVEL \
    --timing-log-option all
"

if [ -n "${HOSTFILE:-}" ]; then
    CLUSTER_MPI_ARGS="
        --hostfile $HOSTFILE \
        --mca plm_rsh_num_concurrent 600 \
        --mca routed_radix 600 \
        --mca btl_tcp_if_include bond0 \
        --mca oob_tcp_if_include bond0 \
        --mca btl_openib_allow_ib false \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_IB_QPS_PER_CONNECTION=8 \
        -x NCCL_IB_TIMEOUT=19 \
        -x NCCL_NET_OVERHEAD=1000 \
    "
fi

mkdir -p logs
set -x
# --num-layers-per-virtual-pipeline-stage $PP_l \
# nsys profile -s none -t nvtx,cuda \
#     --output myrun.nsys-rep \
#     --force-overwrite true \
#     --capture-range=cudaProfilerApi \
#     --capture-range-end=stop \
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
    --use-distributed-optimizer \
    --accumulate-allreduce-grads-in-fp32 \
    --initial-loss-scale 1 \
    --use-flash-attn \
    $PP_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP \
    $CKPT_ARGS \
    ${TP_OVERLAP_ARGS:-} \
    --manual-gc \
    --manual-gc-interval 9999 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TIMING_ARGS \
    2>&1 | tee logs/llama_$TS.txt
# --sequence-parallel \
# --use-mcore-models \