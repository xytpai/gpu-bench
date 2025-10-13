#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

__device__ void global_barrier(int *counter) {
    __shared__ bool is_last_block;
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(counter, 1);
        is_last_block = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last_block) {
        *counter = 0;
    } else {
        while (*reinterpret_cast<int volatile *>(counter) != 0) {}
    }
    __syncthreads();
}

__global__ void global_barrier_test_kernel(int *counter) {
    if (threadIdx.x == 0) {
        printf("[1] from block %d\n", blockIdx.x);
    }
    global_barrier(counter);
    if (threadIdx.x == 0) {
        printf("[2] from block %d\n", blockIdx.x);
    }
    global_barrier(counter);
    if (threadIdx.x == 0) {
        printf("[3] from block %d\n", blockIdx.x);
    }
    global_barrier(counter);
    if (threadIdx.x == 0) {
        printf("[4] from block %d\n", blockIdx.x);
    }
    global_barrier(counter);
}

void global_barrier_test() {
    int *counter;
    gpuMalloc(&counter, 1 * sizeof(int));
    gpuMemset(counter, 0, 1 * sizeof(int));
    dim3 threadsPerBlock(256);
    dim3 numBlocks(64);
    global_barrier_test_kernel<<<numBlocks, threadsPerBlock>>>(counter);
    gpuDeviceSynchronize();
    gpuFree(counter);
}

int main() {
    std::cout << "global barrier test ... ";
    global_barrier_test();
}
