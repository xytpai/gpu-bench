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
        T flags[1];
        flags[0] = (T)rank;
        for (int c = 0; c < nranks; ++c) {
            gpuMemcpy(rs[rank].buffers + c * chunk_size, flags, sizeof(T), gpuMemcpyHostToDevice);
            gpuMemcpy(rs[rank].buffers + (c + 1) * chunk_size - sizeof(T), flags, sizeof(T), gpuMemcpyHostToDevice);
            gpuDeviceSynchronize();
        }
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
        T results[1];
        for (int c = 0; c < nranks; ++c) {
            gpuMemcpy(results, rs[rank].buffers + c * chunk_size, sizeof(T), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            if (results[0] != sum) valid = false;
            gpuMemcpy(results, rs[rank].buffers + (c + 1) * chunk_size - sizeof(T), sizeof(T), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            if (results[0] != sum) valid = false;
            // std::cout << results[0] << " ";
        }
        // std::cout << "\n";
    }
    return valid;
}
