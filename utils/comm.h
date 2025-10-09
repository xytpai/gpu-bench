#pragma once

#include <hip/hip_runtime.h>

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
