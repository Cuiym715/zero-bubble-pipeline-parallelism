#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export TRAIN_ITERS=200

# 为什么这么长
# export SEQ_LENGTH=12288
export SEQ_LENGTH=2048
export GLOBAL_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1

export HOSTFILE=
export MASTER_ADDR=localhost # 127.0.0.1
export NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5,6,7


export TP=1
export CP=1
export PP=$NUM_GPUS
export PP_SCHEDULE=hanayo
export V_SIZE=
export WAVE_SIZE=1
export ZERO_BUBBLE_MEM_LIMIT=$((2*$PP))
export PP_l=1
export CKPT=no

export CUDA_DEVICE_LOG_LEVEL=warn
export TORCH_CUDA_LOG_LEVEL=warn
export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d4-7b-hanayo
export TIMING_LOG_LEVEL=2
export PROFILE_STEP_START=500
export PROFILE_STEP_END=510

rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh
