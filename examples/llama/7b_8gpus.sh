#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export TRAIN_ITERS=50

# 为什么这么长
# export SEQ_LENGTH=12288
export SEQ_LENGTH=2048
export GLOBAL_BATCH_SIZE=16
export MICRO_BATCH_SIZE=2

export HOSTFILE=
export MASTER_ADDR=localhost # 127.0.0.1
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


export TP=1
export CP=1
export PP=$NUM_GPUS
export PP_SCHEDULE=v-zb
export V_SIZE=
export ZERO_BUBBLE_MEM_LIMIT=$((2*$PP))
export PP_l=1
export CKPT=no

export TIMER_SAVE_PATH=/workspace/workspace/model/llama/d4-n8-mbs2-7b-vzb-timerlog
export TIMING_LOG_LEVEL=2
export PROFILE_STEP_START=500
export PROFILE_STEP_END=510

rm -rf $TIMER_SAVE_PATH/profiler_logs
./pretrain_llama.sh
