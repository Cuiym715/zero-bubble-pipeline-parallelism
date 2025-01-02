#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export TRAIN_ITERS=8

# 为什么这么长
# export SEQ_LENGTH=12288
export SEQ_LENGTH=2048
export GLOBAL_BATCH_SIZE=4
export MICRO_BATCH_SIZE=1

export HOSTFILE=
export MASTER_ADDR=localhost # 127.0.0.1
export NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5,6,7


export TP=1
export CP=1
export PP=$NUM_GPUS
export PP_SCHEDULE=gpipe
export V_SIZE=
export WAVE_SIZE=1
export ZERO_BUBBLE_MEM_LIMIT=$((2*$PP))
export PP_l=1
export CKPT=no

export CUDA_DEVICE_LOG_LEVEL=warn
export TORCH_CUDA_LOG_LEVEL=warn
export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d8-mbs1-gpipe
export TIMING_LOG_LEVEL=1
export PROFILE_STEP_START=500
export PROFILE_STEP_END=510

rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh

export GLOBAL_BATCH_SIZE=8
export MICRO_BATCH_SIZE=2
export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d8-mbs2-gpipe
rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh

export GLOBAL_BATCH_SIZE=16
export MICRO_BATCH_SIZE=4
export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d8-mbs4-gpipe
rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh