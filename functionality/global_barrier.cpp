#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

__device__ void global_barrier(int *counter) {
    __shared__ bool is_last_block;
    __threadfence();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(counter, 1);
        is_last_block = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last_block) {
        *counter = 0;
        __threadfence();
    } else {
        while (*reinterpret_cast<int volatile *>(counter) != 0) {}
    }
    __syncthreads();
}

__global__ void global_barrier_test_kernel(int *counter, int *nums) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        nums[bid] = 1;
    }
    global_barrier(counter);
    for (int offset = gridDim.x / 2; offset > 0; offset /= 2) {
        if (tid == 0 && bid < offset) {
            nums[bid] += nums[bid + offset];
        }
        global_barrier(counter);
    }
}

void global_barrier_test() {
    gpuFuncAttributes attr;
    gpuFuncGetAttributes(&attr, (const void *)global_barrier_test_kernel);
    int registers_per_thread = attr.numRegs;
    int regs_per_block;
    int device{-1};
    gpuGetDevice(&device);
    gpuDeviceGetAttribute(&regs_per_block, gpuDevAttrMaxRegistersPerBlock, device);
    int max_threads_per_block = std::min(regs_per_block / registers_per_thread, 1024);
    int sm_count;
    gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount, device);
    std::cout << "registers_per_thread: " << registers_per_thread << "\n";
    std::cout << "regs_per_block: " << regs_per_block << "\n";
    std::cout << "max_threads_per_block: " << max_threads_per_block << "\n";
    std::cout << "sm_count: " << sm_count << "\n";
    int block_size = max_threads_per_block / 4;
    std::cout << "block_size: " << block_size << "\n";
    int nblocks = sm_count;
    std::cout << "nblocks: " << nblocks << "\n";
    int *counter, *nums;
    gpuMalloc(&counter, 1 * sizeof(int));
    gpuMemset(counter, 0, 1 * sizeof(int));
    gpuMalloc(&nums, nblocks * sizeof(int));
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(nblocks);
    global_barrier_test_kernel<<<numBlocks, threadsPerBlock>>>(counter, nums);
    gpuDeviceSynchronize();
    int nums_;
    gpuMemcpy(&nums_, nums, 1 * sizeof(int), gpuMemcpyDeviceToHost);
    std::cout << nums_ << "\n";
    assert(nums_ == nblocks);
    gpuFree(counter);
    gpuFree(nums);
}

int main() {
    std::cout << "global barrier test ... \n";
    global_barrier_test();
    std::cout << "ok\n";
}
