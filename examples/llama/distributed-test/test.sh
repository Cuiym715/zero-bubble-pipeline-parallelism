HOSTFILE=/workspace/zero-bubble-pipeline-parallelism/examples/llama/hostfile

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL mpirun --allow-run-as-root \
    --mca plm_rsh_args "-p 2222" \
    --hostfile $HOSTFILE \
    --wdir /workspace/zero-bubble-pipeline-parallelism/examples/llama/distributed-test \
    -np 8 \
    ./test_nccl