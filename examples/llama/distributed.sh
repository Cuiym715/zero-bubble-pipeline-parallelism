#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL

export TRAIN_ITERS=20

# 为什么这么长
# export SEQ_LENGTH=12288
export SEQ_LENGTH=2048
export GLOBAL_BATCH_SIZE=16
export MICRO_BATCH_SIZE=1

export HOSTFILE=/workspace/zero-bubble-pipeline-parallelism/examples/llama/hostfile
export MASTER_ADDR=192.168.0.91
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=4,5,6,7


export TP=1
export CP=1
export PP=$NUM_GPUS
export PP_SCHEDULE=1f1b
export V_SIZE=
export WAVE_SIZE=1
export ZERO_BUBBLE_MEM_LIMIT=$((2*$PP))
export PP_l=1
export CKPT=no

export CUDA_DEVICE_LOG_LEVEL=warn
export TORCH_CUDA_LOG_LEVEL=warn
export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d8-mbs1-1f1b
export TIMING_LOG_LEVEL=1
export PROFILE_STEP_START=500
export PROFILE_STEP_END=510

rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh

# export GLOBAL_BATCH_SIZE=32
# export MICRO_BATCH_SIZE=2
# export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d8-mbs2-vzb-ts
# rm -rf $TIMER_SAVE_PATH/profiler_logs
# ./pretrain_llama.sh