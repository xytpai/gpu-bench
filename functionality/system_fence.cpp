#include <iostream>
#include <random>
#include <cassert>
#include <vector>
#include <tuple>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#else
#include <cuda_runtime.h>
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#endif

int enable_p2p() {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        for (int peer = 0; peer < ngpus; ++peer) {
            if (local == peer) continue;
            int can = 0;
            gpuDeviceCanAccessPeer(&can, local, peer);
            assert(can);
            gpuDeviceEnablePeerAccess(peer, 0);
        }
    }
    return ngpus;
}

__global__ void produce_kernel(int *data, int *flag, int n, int base, bool use_fence) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
        data[idx] = base + idx; // set data
    }
    if (use_fence) __threadfence_system();
    *flag = 1;
}

__global__ void consume_kernel(int *data, int *flag, int n, int base) {
    volatile bool done = false;
    while(!done) {
        done = *flag != 0;
    }
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
        if(data[idx] != base + idx) {
            printf("error\n");
        }
    }
}

void threadfence_system_test(int dev_p, int dev_c, int n) {
    static int base = 0;
    int *data, *flag;

    gpuGetDevice(&dev_p);
    gpuMalloc(&data, n * sizeof(int));
    gpuMemset(data, 0, n * sizeof(int));
    gpuDeviceSynchronize();

    gpuGetDevice(&dev_c);
    gpuMalloc(&flag, 1 * sizeof(int));
    gpuMemset(flag, 0, 1 * sizeof(int));
    gpuDeviceSynchronize();

    dim3 threadsPerBlock(256);
    dim3 numBlocks(256);

    gpuGetDevice(&dev_c);
    consume_kernel<<<numBlocks, threadsPerBlock>>>(data, flag, n, base);
    gpuGetDevice(&dev_p);
    produce_kernel<<<numBlocks, threadsPerBlock>>>(data, flag, n, base, true);
    
    gpuGetDevice(&dev_p);
    gpuDeviceSynchronize();
    gpuGetDevice(&dev_c);
    gpuDeviceSynchronize();

    gpuGetDevice(&dev_p);
    gpuFree(data);
    gpuGetDevice(&dev_c);
    gpuFree(flag);
    base++;
}

int main() {
    enable_p2p();
    std::cout << "system fence test ... \n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "step:" << i << "\n";
        threadfence_system_test(0, 1, 1024*4);
    }
    std::cout << "ok\n";
}
