#pragma once

#include "collectives.h"

template <typename T>
void reset_reduce_flags(std::vector<GPUResources> &rs) {
    auto element_bytes = sizeof(T);
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        auto flags = new T[chunk_len];
        for (auto i = 0; i < chunk_len; ++i) {
            flags[i] = (T)rank;
        }
        for (int c = 0; c < nranks; ++c) {
            gpuMemcpy(rs[rank].buffers + c * chunk_size, flags, chunk_size, gpuMemcpyHostToDevice);
            gpuDeviceSynchronize();
        }
        delete[] flags;
    }
}

template <typename T>
bool validate_reduce_flags(std::vector<GPUResources> &rs) {
    auto element_bytes = sizeof(T);
    int nranks = rs.size();
    int sum = (nranks - 1) * nranks / 2;
    bool valid = true;
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        auto results = new T[chunk_len];
        for (int c = 0; c < nranks; ++c) {
            gpuMemcpy(results, rs[rank].buffers + c * chunk_size, chunk_size, gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            for (auto i = 0; i < chunk_len; ++i) {
                if (results[i] != sum) valid = false;
            }
            // std::cout << results[0] << " ";
        }
        delete[] results;
        // std::cout << "\n";
    }
    return valid;
}
