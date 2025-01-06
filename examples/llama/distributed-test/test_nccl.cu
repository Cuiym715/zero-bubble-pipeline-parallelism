#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    }

#define NCCL_CHECK(call) \
    { \
        ncclResult_t result = call; \
        if (result != ncclSuccess) { \
            printf("NCCL Error: %s:%d, ", __FILE__, __LINE__); \
            printf("reason: %s\n", ncclGetErrorString(result)); \
            exit(1); \
        } \
    }

int main(int argc, char* argv[]) {
    const int numGPUs = 2; // Adjust for the number of GPUs
    int devs[numGPUs] = {0, 1}; // Replace with your device IDs
    cudaStream_t streams[numGPUs];
    ncclComm_t comms[numGPUs];
    float *sendbuff[numGPUs], *recvbuff[numGPUs];
    size_t size = 1024 * 1024;

    printf("Starting NCCL Test Program\n");

    // Initialize CUDA streams and memory
    for (int i = 0; i < numGPUs; i++) {
        printf("Setting up device %d\n", devs[i]);
        CHECK(cudaSetDevice(devs[i]));
        CHECK(cudaMalloc(&sendbuff[i], size * sizeof(float)));
        CHECK(cudaMalloc(&recvbuff[i], size * sizeof(float)));
        CHECK(cudaStreamCreate(&streams[i]));
    }

    printf("Initializing NCCL\n");

    // Initialize NCCL
    NCCL_CHECK(ncclCommInitAll(comms, numGPUs, devs));

    printf("NCCL initialized. Starting AllReduce\n");

    // Perform AllReduce
    for (int i = 0; i < numGPUs; i++) {
        printf("Performing AllReduce on device %d\n", devs[i]);
        CHECK(cudaSetDevice(devs[i]));
        NCCL_CHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum, comms[i], streams[i]));
    }

    printf("Waiting for streams to finish\n");

    // Synchronize and cleanup
    for (int i = 0; i < numGPUs; i++) {
        printf("Synchronizing stream on device %d\n", devs[i]);
        CHECK(cudaStreamSynchronize(streams[i]));
        CHECK(cudaFree(sendbuff[i]));
        CHECK(cudaFree(recvbuff[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }

    printf("NCCL AllReduce completed successfully!\n");
    return 0;
}
