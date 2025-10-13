#pragma once

#include <cassert>
#include <vector>
#include <tuple>

#include "device_common.h"

int enable_p2p() {
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
    return ngpus;
}

struct GPUResources {
    size_t buffer_size;
    size_t chunk_size;
    int num_chunks;
    int num_streams;
    std::vector<std::vector<unsigned char *>> buffers;
    std::vector<gpuStream_t> streams;
    // barrier
    std::vector<int *> barrier_flags;
    int *counter;
    int *flag;
};

#define DEFAULT_P2P_CHUNK_SIZE (1024 * 1024 * 32)

void allocate_resources(std::vector<GPUResources> &rs, size_t buffer_size, size_t chunk_size, int streams_per_gpu, int nblocks_per_gpu = 256) {
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
        rs[rank].barrier_flags.resize(ngpus);
        for (int peer = 0; peer < ngpus; ++peer) {
            gpuMalloc(&rs[rank].barrier_flags[peer], ngpus * nblocks_per_gpu * sizeof(int));
        }
        gpuMalloc(&rs[rank].counter, sizeof(int));
        gpuMalloc(&rs[rank].flag, sizeof(int));
    }
}

void delete_resources(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int rank = 0; rank < ngpus; ++rank) {
        gpuSetDevice(rank);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
        for (int peer = 0; peer < ngpus; ++peer) {
            for (auto p : rs[rank].buffers[peer]) gpuFree(p);
            gpuFree(rs[rank].barrier_flags[peer]);
        }
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
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

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (*reinterpret_cast<int volatile *>(counter_ptr) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
class Barrier {
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const &comm) {
        if (threadIdx.x < NRanks) {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int *>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag = reinterpret_cast<int *>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        constexpr int kBarrierFlagCount = 256;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            m_flag_value = next_flag(m_flag_value);
            // To avoid the ABA problem, we need to synchronize the correct flag value to all
            // barrier_flags, even if the corresponding CTA has not been launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x) {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
            }
        }
        __syncthreads();
    }

protected:
    __device__ __forceinline__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        auto ptr = reinterpret_cast<volatile uint32_t *>(addr);
        auto flag_ = *reinterpret_cast<volatile uint32_t *>(&flag);
        __atomic_store_n(ptr, flag_, __ATOMIC_SEQ_CST);
#endif
    }

    __device__ __forceinline__ int ld_flag(int *addr) {
        int flag;
#ifdef __CUDACC__
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                     : "=r"(flag)
                     : "l"(addr));
#else
        auto ptr = reinterpret_cast<volatile uint32_t *>(addr);
        *reinterpret_cast<volatile uint32_t *>(&flag) = __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
#endif
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag) {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag) {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};
