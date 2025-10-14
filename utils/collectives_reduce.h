#pragma once

#include "collectives.h"

void allocate_reduce_resources(std::vector<GPUResources> &rs, size_t data_bytes, int streams_per_gpu = 1, int nblocks_per_gpu = DEFAULT_NCTAS) {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    rs.resize(ngpus);
    assert(data_bytes % (ngpus * 4) == 0);
    size_t chunk_size = (data_bytes + (size_t)ngpus - (size_t)1) / (size_t)ngpus;
    size_t rounded_data_bytes = chunk_size * (size_t)ngpus;
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        rs[rank].buffer_size = rounded_data_bytes;
        rs[rank].chunk_size = chunk_size;
        rs[rank].num_chunks = ngpus;
        rs[rank].buffers.resize(ngpus);
        for (int c = 0; c < ngpus; ++c) {
            rs[rank].buffers[c].resize(1);
            gpuMalloc(&rs[rank].buffers[c][0], chunk_size);
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

void delete_reduce_resources(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
        for (int c = 0; c < ngpus; ++c) {
            gpuFree(rs[rank].buffers[c][0]);
        }
        gpuFree(rs[rank].barrier_flags);
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
    }
}

template <typename T>
void reset_reduce_flags(std::vector<GPUResources> &rs) {
    auto element_bytes = sizeof(T);
    for (int rank = 0; rank < rs.size(); ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        int num_chunks = rs[rank].num_chunks;
        auto flags = new T[chunk_len];
        for (auto i = 0; i < chunk_len; ++i) {
            flags[i] = (T)rank;
        }
        for (int c = 0; c < num_chunks; ++c) {
            gpuMemcpy(rs[rank].buffers[c][0], flags, chunk_size, gpuMemcpyHostToDevice);
            gpuDeviceSynchronize();
        }
        delete[] flags;
    }
}

template <typename T>
bool validate_reduce_flags(std::vector<GPUResources> &rs) {
    auto element_bytes = sizeof(T);
    int ngpus = rs.size();
    int sum = (ngpus - 1) * ngpus / 2;
    bool valid = true;
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        size_t chunk_size = rs[local].chunk_size;
        assert(chunk_size % element_bytes == 0);
        size_t chunk_len = chunk_size / element_bytes;
        int num_chunks = rs[local].num_chunks;
        auto results = new T[chunk_len];
        for (int c = 0; c < num_chunks; ++c) {
            gpuMemcpy(results, rs[local].buffers[c][0], chunk_size, gpuMemcpyDeviceToHost);
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
