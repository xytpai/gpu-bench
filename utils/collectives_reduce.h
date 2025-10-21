#pragma once

#include "collectives.h"

template <typename T>
void reset_reduce_flags(std::vector<GPUResources> &rs, bool strict = true) {
    auto element_bytes = sizeof(T);
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        if (!strict) {
            T flags[1];
            flags[0] = (T)rank;
            for (int c = 0; c < nranks; ++c) {
                gpuMemcpy(rs[rank].buffers + c * chunk_size, flags, sizeof(T), gpuMemcpyHostToDevice);
                gpuMemcpy(rs[rank].buffers + (c + 1) * chunk_size - sizeof(T), flags, sizeof(T), gpuMemcpyHostToDevice);
                gpuDeviceSynchronize();
            }
        } else {
            auto flags = new T[chunk_len];
            for (int i = 0; i < chunk_len; ++i) flags[i] = (T)rank;
            for (int c = 0; c < nranks; ++c) {
                gpuMemcpy(rs[rank].buffers + c * chunk_size, flags, chunk_size, gpuMemcpyHostToDevice);
                gpuDeviceSynchronize();
            }
            delete[] flags;
        }
    }
}

template <typename T>
bool validate_reduce_flags(std::vector<GPUResources> &rs, bool strict = false) {
    auto element_bytes = sizeof(T);
    int nranks = rs.size();
    int sum = (nranks - 1) * nranks / 2;
    bool valid = true;
    auto target = sum;
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        if (!strict) {
            T results[2];
            for (int c = 0; c < nranks; ++c) {
                gpuMemcpy(results, rs[rank].buffers + c * chunk_size, sizeof(T), gpuMemcpyDeviceToHost);
                gpuDeviceSynchronize();
                if (std::abs(results[0] - target) / target > 0.01) valid = false;
                gpuMemcpy(results + 1, rs[rank].buffers + (c + 1) * chunk_size - sizeof(T), sizeof(T), gpuMemcpyDeviceToHost);
                gpuDeviceSynchronize();

                if (std::abs(results[1] - target) / target > 0.01) valid = false;
                std::cout << results[1] << " ";
            }
            std::cout << "\n";
        } else {
            auto results = new T[chunk_len];
            for (int c = 0; c < nranks; ++c) {
                gpuMemcpy(results, rs[rank].buffers + c * chunk_size, chunk_size, gpuMemcpyDeviceToHost);
                gpuDeviceSynchronize();
                for (int i = 0; i < chunk_len; ++i) {
                    if (std::abs(results[i] - target) / target > 0.01) valid = false;
                }
            }
            delete[] results;
        }
    }
    return valid;
}
