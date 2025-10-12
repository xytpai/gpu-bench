#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

template <typename T, int vec_size, int loops>
__global__ void threads_incre_kernel(T *local, T *peer, const size_t n) {
    const int block_work_size = loops * blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    auto remaining = n - index;
#pragma unroll
    for (int i = 0; i < loops; ++i) {
        if (remaining < vec_size) {
            for (auto i = index; i < n; i++) {
                local[i] = peer[i] + 1;
                peer[i] = local[i];
            }
        } else {
            using vec_t = aligned_array<T, vec_size>;
            auto peer_vec_ = reinterpret_cast<vec_t *>(&peer[index]);
            auto local_vec_ = reinterpret_cast<vec_t *>(&local[index]);
            auto peer_vec = *peer_vec_;
            for (int v = 0; v < vec_size; ++v) {
                peer_vec.val[v] += 1;
            }
            *local_vec_ = peer_vec;
            *peer_vec_ = *local_vec_;
        }
        index += blockDim.x * vec_size;
    }
}

template <typename T, int vec_size, int loops>
float threads_incre(T *local, T *peer, size_t n) {
    const int block_size = 256;
    const int block_work_size = loops * block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);

    threads_incre_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock>>>(local, peer, n);
    gpuDeviceSynchronize();

    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);
    return ms;
}

float measure_uva_access(int local, int peer, const size_t n, bool val) {
    auto bytes = n * sizeof(float);
    auto data = new float[n];
    for (int i = 0; i < n; i++)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *local_ptr, *peer_ptr;

    gpuSetDevice(local);
    gpuMalloc(&local_ptr, bytes);
    gpuMemcpy(local_ptr, data, n * sizeof(float), gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();

    gpuSetDevice(peer);
    gpuMalloc(&peer_ptr, bytes);
    gpuMemcpy(peer_ptr, data, n * sizeof(float), gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();

    float timems;
    int iters = 2;
    for (int i = 0; i < iters; i++)
        timems = threads_incre<float, 4, 1>(local_ptr, peer_ptr, n);

    float total_GBytes = (n + n) * sizeof(float);
    float gbps = total_GBytes / (timems) / 1000.0 / 1000.0;

    if (val) {
        auto out_data = new float[n];
        gpuSetDevice(peer);
        gpuMemcpy(out_data, peer_ptr, n * sizeof(float), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        for (int i = 0; i < n; i++) {
            auto diff = (float)out_data[i] - iters - (float)data[i];
            diff = diff > 0 ? diff : -diff;
            if (diff > 0.01) {
                std::cout << "error\n";
                return -1;
            }
        }
        delete[] out_data;
    }

    gpuFree(local_ptr);
    gpuFree(peer_ptr);
    delete[] data;
    return gbps;
}

int main() {
    std::cout << "1GB threads uva peer copy test <bi-dir> ... (GBps)\n";
    enable_p2p();
    int device_count = 0;
    gpuGetDeviceCount(&device_count);
    std::cout << std::right << std::setw(11) << "local-peer";
    for (int j = 0; j < device_count; ++j) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
    }
    std::cout << "\n";
    for (int local = 0; local < device_count; ++local) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(local) + "]");
        for (int peer = 0; peer < device_count; ++peer) {
            auto bw = measure_uva_access(local, peer, 1024 * 1024 * 256 + 2, true);
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
        }
        std::cout << "\n";
    }
}
