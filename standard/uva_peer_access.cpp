#include <iostream>
#include <random>

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

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);

    threads_copy_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock>>>(in, out, n);
    gpuDeviceSynchronize();

    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);
    return ms;
}

void measure_uva_access(const size_t n) {
    auto bytes = n * sizeof(float);
    auto data = new float[n];
    for (int i = 0; i < n; i++)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *local = nullptr;
    gpuSetDevice(0);
    gpuMalloc(&local, bytes);
    gpuMemcpy(local, data, n * sizeof(float), gpuMemcpyHostToDevice);

    float *peer = nullptr;
    gpuSetDevice(1);
    gpuMalloc(&peer, bytes);
    gpuMemset(peer, 0, bytes);

    float timems;
    for (int i = 0; i < 2; i++)
        timems = threads_copy<float, 4, 1>(local, peer, n);

    std::cout << "timems:" << timems << " throughput:";
    float total_GBytes = (n + n) * sizeof(float) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS val:";

    auto out_data = new float[n];
    gpuMemcpy(out_data, peer, n * sizeof(float), gpuMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        auto diff = (float)out_data[i] - (float)data[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    gpuFree(peer);
    gpuFree(local);
    delete[] data;
    delete[] out_data;
}

int main() {
    std::cout << "1GB threads uva peer copy test ...\n";
    measure_uva_access(1024 * 1024 * 256 + 2);
}
