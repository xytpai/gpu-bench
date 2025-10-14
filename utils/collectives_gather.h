#pragma once

#include "collectives.h"

void allocate_gather_resources(std::vector<GPUResources> &rs, size_t buffer_size, size_t chunk_size, int streams_per_gpu, int nblocks_per_gpu = DEFAULT_NCTAS) {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    rs.resize(ngpus);
    int num_chunks = (int)((buffer_size + chunk_size - 1) / chunk_size);
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        rs[rank].buffer_size = buffer_size;
        rs[rank].chunk_size = chunk_size;
        rs[rank].num_chunks = num_chunks;
        rs[rank].buffers.resize(ngpus);
        for (int peer = 0; peer < ngpus; ++peer) {
            rs[rank].buffers[peer].resize(num_chunks);
            for (int c = 0; c < num_chunks; ++c) {
                gpuMalloc(&rs[rank].buffers[peer][c], chunk_size);
            }
        }
        rs[rank].num_streams = streams_per_gpu;
        rs[rank].streams.resize(streams_per_gpu);
        for (int s = 0; s < streams_per_gpu; ++s) {
            gpuStreamCreate(&rs[rank].streams[s]);
        }
        // barrier
        gpuMalloc(&rs[rank].barrier_flags, nblocks_per_gpu * ngpus * sizeof(int));
        gpuMalloc(&rs[rank].counter, sizeof(int));
        gpuMalloc(&rs[rank].flag, sizeof(int));
        rs[rank].nblocks = nblocks_per_gpu;
    }
}

void delete_gather_resources(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
        for (int peer = 0; peer < ngpus; ++peer) {
            for (auto p : rs[rank].buffers[peer]) gpuFree(p);
        }
        gpuFree(rs[rank].barrier_flags);
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
    }
}

void reset_gather_flags(std::vector<GPUResources> &rs, unsigned char flag) {
    for (int rank = 0; rank < rs.size(); ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        int num_chunks = rs[rank].num_chunks;
        unsigned char start_flag = (flag + rank) % 255;
        unsigned char end_flag = (flag + rank + 1) % 255;
        gpuMemset(rs[rank].buffers[rank][0], start_flag, 1);
        gpuMemset(rs[rank].buffers[rank][num_chunks - 1] + chunk_size - 1, end_flag, 1);
        gpuDeviceSynchronize();
    }
}

bool validate_gather_flags(std::vector<GPUResources> &rs, unsigned char flag, std::vector<std::vector<bool>> &mask) {
    auto data = new unsigned char[2];
    int ngpus = rs.size();
    bool c0, c1;
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        size_t chunk_size = rs[local].chunk_size;
        int num_chunks = rs[local].num_chunks;
        for (int peer = 0; peer < ngpus; ++peer) {
            if (!mask[local][peer]) continue;
            unsigned char start_flag = (flag + peer) % 255;
            unsigned char end_flag = (flag + peer + 1) % 255;
            gpuMemcpy(data, rs[local].buffers[peer][0], 1, gpuMemcpyDeviceToHost);
            gpuMemcpy(data + 1, rs[local].buffers[peer][num_chunks - 1] + chunk_size - 1, 1, gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            c0 = data[0] == start_flag;
            c1 = data[1] == end_flag;
            if (!(c0 && c1)) return false;
        }
    }
    delete[] data;
    return true;
}
