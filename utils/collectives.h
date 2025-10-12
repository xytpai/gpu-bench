#pragma once

#include <cassert>
#include <vector>
#include <tuple>

#include "device_common.h"
#include "memory.h"

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
void threads_copy(const T *in, T *out, size_t n, gpuStream_t s) {
    const int block_size = 256;
    const int block_work_size = loops * block_size * vec_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);
    threads_copy_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock, 0, s>>>(in, out, n);
}

void memcpy_peer_async(unsigned char *dst, int dst_dev, unsigned char *src, int src_dev, size_t n, gpuStream_t s, bool use_p2p) {
    if (use_p2p) {
        gpuMemcpyPeerAsync(dst, dst_dev, src, src_dev, n, s);
    } else {
        gpuSetDevice(dst_dev);
        threads_copy<unsigned char, 16, 1>(src, dst, n, s);
    }
}

void enable_p2p() {
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
}

struct GPUResources {
    size_t buffer_size;
    size_t chunk_size;
    int num_chunks;
    int num_streams;
    std::vector<std::vector<unsigned char *>> buffers;
    std::vector<gpuStream_t> streams;
};

#define DEFAULT_P2P_CHUNK_SIZE (1024 * 1024 * 32)

void allocate_resources(std::vector<GPUResources> &rs, size_t buffer_size, size_t chunk_size, int streams_per_gpu) {
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
    }
}

void delete_resources(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
        for (int peer = 0; peer < ngpus; ++peer) {
            for (auto p : rs[rank].buffers[peer]) gpuFree(p);
        }
    }
}

std::tuple<unsigned char, unsigned char> get_flag(int rank, unsigned char flag) {
    unsigned char start = (flag + rank) % 255;
    unsigned char end = (flag + rank + 1) % 255;
    return {start, end};
}

void reset_gather_flags(std::vector<GPUResources> &rs, unsigned char flag) {
    for (int rank = 0; rank < rs.size(); ++rank) {
        gpuSetDevice(rank);
        size_t chunk_size = rs[rank].chunk_size;
        int num_chunks = rs[rank].num_chunks;
        auto [start_flag, end_flag] = get_flag(rank, flag);
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
            auto [start_flag, end_flag] = get_flag(peer, flag);
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
