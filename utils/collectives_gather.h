#pragma once

#include "collectives.h"

void reset_gather_flags(std::vector<GPUResources> &rs) {
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        unsigned char start_flag = rank % 255;
        unsigned char end_flag = (rank + 1) % 255;
        gpuMemset(rs[rank].buffers + rank * chunk_size, start_flag, 1);
        gpuMemset(rs[rank].buffers + (rank + 1) * chunk_size - 1, end_flag, 1);
        gpuDeviceSynchronize();
    }
}

bool validate_gather_flags(std::vector<GPUResources> &rs, std::vector<std::vector<bool>> &mask) {
    auto data = new unsigned char[2];
    int nranks = rs.size();
    bool c0, c1;
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        for (int peer = 0; peer < nranks; ++peer) {
            if (!mask[rank][peer]) continue;
            unsigned char start_flag = peer % 255;
            unsigned char end_flag = (peer + 1) % 255;
            gpuMemcpy(data, rs[rank].buffers + peer * chunk_size, 1, gpuMemcpyDeviceToHost);
            gpuMemcpy(data + 1, rs[rank].buffers + (peer + 1) * chunk_size - 1, 1, gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            c0 = data[0] == start_flag;
            c1 = data[1] == end_flag;
            if (!(c0 && c1)) return false;
        }
    }
    delete[] data;
    return true;
}
