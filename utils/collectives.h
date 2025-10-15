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
    size_t chunk_size;
    unsigned char *buffers;
    int num_streams;
    std::vector<gpuStream_t> streams;
    size_t segment_size;
    // barrier
    int nblocks;
    int *barrier_flags;
    int *counter;
    int *flag;
};

#define DEFAULT_NCTAS 256

void allocate_resources(std::vector<GPUResources> &rs, size_t chunk_size, size_t segment_size, int streams_per_gpu, int nblocks_per_gpu = DEFAULT_NCTAS) {
    int nranks = 0;
    gpuGetDeviceCount(&nranks);
    rs.resize(nranks);
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        rs[rank].chunk_size = chunk_size;
        rs[rank].segment_size = segment_size;
        gpuMalloc(&rs[rank].buffers, nranks * chunk_size);
        rs[rank].num_streams = streams_per_gpu;
        rs[rank].streams.resize(streams_per_gpu);
        for (int s = 0; s < streams_per_gpu; ++s) {
            gpuStreamCreate(&rs[rank].streams[s]);
        }
        // barrier
        gpuMalloc(&rs[rank].barrier_flags, nblocks_per_gpu * nranks * sizeof(int));
        gpuMalloc(&rs[rank].counter, sizeof(int));
        gpuMalloc(&rs[rank].flag, sizeof(int));
        rs[rank].nblocks = nblocks_per_gpu;
    }
}

void delete_resources(std::vector<GPUResources> &rs) {
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuFree(rs[rank].buffers);
        gpuFree(rs[rank].barrier_flags);
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
    }
}

class GPUWorkSpace {
public:
    GPUWorkSpace() :
        workspace_(nullptr) {
    }
    void init(std::vector<GPUResources> &rs, int rank) {
        gpuSetDevice(rank);
        int nranks = rs.size();
        auto &r = rs[rank];
        gpuMemset(r.barrier_flags, 0, r.nblocks * nranks * sizeof(int));
        gpuMemset(r.counter, 0, sizeof(int));
        gpuMemset(r.flag, 0, sizeof(int));
        std::vector<void *> workspace(nranks * 3 + 2);
        for (int peer = 0; peer < nranks; ++peer) {
            workspace[peer] = (void *)rs[peer].buffers;
            workspace[nranks + peer] = (void *)rs[peer].barrier_flags;
        }
        workspace[nranks * 3 + 0] = (void *)r.counter;
        workspace[nranks * 3 + 1] = (void *)r.flag;
        gpuMalloc(&workspace_, (nranks * 3 + 2) * sizeof(void *));
        gpuMemcpy(workspace_, workspace.data(), workspace.size() * sizeof(void *), gpuMemcpyHostToDevice);
    }
    ~GPUWorkSpace() {
        gpuFree(workspace_);
    }
    void **workspace() const {
        return workspace_;
    }

private:
    void **workspace_;
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            current_comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
            next_comm_bufs[r] = workspace[2 * NRanks + r];
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
    void *current_comm_bufs[NRanks];
    void *next_comm_bufs[NRanks];
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
        constexpr int kBarrierFlagCount = DEFAULT_NCTAS;
        __syncthreads();
        __threadfence_system();
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
        __atomic_store_n(addr, flag, __ATOMIC_SEQ_CST);
#endif
    }

    __device__ __forceinline__ int ld_flag(int *addr) {
        int flag;
#ifdef __CUDACC__
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                     : "=r"(flag)
                     : "l"(addr));
#else
        flag = __atomic_load_n(addr, __ATOMIC_SEQ_CST);
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
