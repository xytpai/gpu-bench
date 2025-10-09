#include <iostream>
#include <random>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "utils.h"
using namespace std;

template <typename T, int vec_size, int loops>
__global__ void threads_copy_kernel(const T *in, T *out, const size_t n) {
    const int block_work_size = loops * blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    auto remaining = n - index;
#pragma unroll
    for (int i = 0; i < loops; ++i) {
        if (remaining < vec_size) {
            for (auto i = index; i < n; i++) {
                out[i] = in[i];
            }
        } else {
            using vec_t = aligned_array<T, vec_size>;
            auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
            auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
            *out_vec = *in_vec;
        }
        index += blockDim.x * vec_size;
    }
}

template <typename T, int vec_size, int loops>
float threads_copy(const T *in, T *out, size_t n) {
    const int block_size = 256;
    const int block_work_size = loops * block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    threads_copy_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock>>>(in, out, n);
    hipDeviceSynchronize();

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    return ms;
}

template <int vec_size, typename scalar_t, int loops>
void test_threads_copy(size_t n) {
    auto in_cpu = new scalar_t[n];
    auto out_cpu = new scalar_t[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    scalar_t *in_hip, *out_hip;
    hipMalloc(&in_hip, n * sizeof(scalar_t));
    hipMalloc(&out_hip, n * sizeof(scalar_t));

    hipMemcpy(in_hip, in_cpu, n * sizeof(scalar_t), hipMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 2; i++)
        timems = threads_copy<scalar_t, vec_size, loops>(in_hip, out_hip, n);
    std::cout << "timems:" << timems << " throughput:";

    float total_GBytes = (n + n) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS val:";

    hipMemcpy(out_cpu, out_hip, n * sizeof(scalar_t), hipMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        auto diff = (float)out_cpu[i] - (float)in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    hipFree(in_hip);
    hipFree(out_hip);
    delete[] in_cpu;
    delete[] out_cpu;
}

int main() {
    constexpr int loops = 1;
    std::cout << "1GB threads copy test ...\n";
    std::cout << "float1: ";
    test_threads_copy<1, float, loops>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_copy<2, float, loops>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_copy<4, float, loops>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_copy<8, float, loops>(1024 * 1024 * 256 + 2);
    std::cout << "half1: ";
    test_threads_copy<1, __half, loops>((1024 * 1024 * 256 + 2) * 2);
    std::cout << "half2: ";
    test_threads_copy<2, __half, loops>((1024 * 1024 * 256 + 2) * 2);
    std::cout << "half4: ";
    test_threads_copy<4, __half, loops>((1024 * 1024 * 256 + 2) * 2);
    std::cout << "half8: ";
    test_threads_copy<8, __half, loops>((1024 * 1024 * 256 + 2) * 2);
}
