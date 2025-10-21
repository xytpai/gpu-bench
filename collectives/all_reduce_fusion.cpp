#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;
namespace cg = cooperative_groups;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val, T *shared, const int tid) {
    const int w_tid = tid & 31;
    const int wid = tid >> 5;
    val = warp_reduce_sum(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0) {
        val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

template <int NRanks, typename T>
__global__ void all_reduce_norm_kernel(void **workspace, int rank, size_t count, double *global_acc) {
    SyncComm<NRanks> comm(workspace);
    cg::grid_group grid = cg::this_grid();

    const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * NRanks);
    const int globalNthreads = blockDim.x * gridDim.x * NRanks;
    for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
        T v = (T)0;
        for (int peer = 0; peer < NRanks; ++peer) {
            v += reinterpret_cast<T *>(comm.comm_bufs[peer])[offset];
        }
        for (int peer = 0; peer < NRanks; ++peer) {
            *(reinterpret_cast<T *>(comm.comm_bufs[peer]) + offset) = v;
        }
    }

    // post ops
    Barrier<NRanks> barrier(rank, comm);
    barrier.sync();
    grid.sync();
    const int localTid = threadIdx.x + blockDim.x * blockIdx.x;
    const int localNthreads = blockDim.x * gridDim.x;
    double sum = 0;
    for (size_t offset = localTid; offset < count; offset += localNthreads) {
        sum += reinterpret_cast<T *>(comm.comm_bufs[rank])[offset];
    }
    double __shared__ shared[512];
    sum = block_reduce_sum<double>(sum, shared, threadIdx.x);
    atomicAdd(&global_acc[blockIdx.x], sum);
    barrier.sync();
    grid.sync();
    double __shared__ all_sum;
    if (threadIdx.x == 0) {
        all_sum = global_acc[blockIdx.x];
    }
    __syncthreads();
    all_sum /= count;
    // for (size_t offset = localTid; offset < count; offset += localNthreads) {
    //     *(reinterpret_cast<T *>(comm.comm_bufs[rank]) + offset) = (T)all_sum;
    // }
    comm.update(barrier.m_flag_value);
}

template <typename T>
class AllReduceDirect {
public:
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        int chunk_size = rs[0].chunk_size;
        size_t count = (size_t)chunk_size * nranks / sizeof(T);
        std::vector<GPUWorkSpace> workspaces(nranks);
        double *global_acc[16];
        for (int rank = 0; rank < nranks; ++rank) {
            workspaces[rank].init(rs, rank);
            gpuMalloc(&global_acc[rank], DEFAULT_NCTAS * sizeof(double));
            gpuMemset(global_acc[rank], 0, DEFAULT_NCTAS * sizeof(double));
        }
        for (int rank = 0; rank < nranks; ++rank) {
            gpuSetDevice(rank);
            dim3 threadsPerBlock(256);
            dim3 numBlocks(DEFAULT_NCTAS);
            void **ptr = workspaces[rank].workspace();
            void *args[] = {(void *)&ptr, &rank, &count, &global_acc[rank]};

            switch (nranks) {
            case 8: {
                gpuLaunchCooperativeKernel(all_reduce_norm_kernel<8, T>, numBlocks, threadsPerBlock, args, 2048, nullptr);
            } break;
            case 4: {
                // all_reduce_norm_kernel<4, T><<<numBlocks, threadsPerBlock>>>(
                //     workspaces[rank].workspace(), rank, count, global_acc[rank]);
            } break;
            default:
                return;
            }
        }
        for (int g = 0; g < nranks; ++g) {
            gpuSetDevice(g);
            gpuDeviceSynchronize();
        }
    }
};

template <typename T, typename func_t>
std::tuple<double, bool, double> runbench(int nranks, func_t fn, size_t data_bytes) {
    std::vector<GPUResources> rs;
    assert(data_bytes % nranks == 0);
    size_t chunk_size = data_bytes / nranks;
    allocate_resources(rs, chunk_size, chunk_size, 1);
    for (int w = 0; w < 2; ++w) {
        fn(rs);
    }
    reset_reduce_flags<T>(rs);
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    bool valid = validate_reduce_flags<T>(rs);
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (nranks - 1) * 2 * nranks * chunk_size;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    delete_resources(rs);
    return {gbps, valid, seconds};
}

int main() {
    int nranks = enable_p2p();
    std::cout << "nranks: " << nranks << "\n";
    size_t data_size = (size_t)1024 * 1024 * 1024;
    {
        std::cout << "======== 1GB all reduce direct test ========\n";
        using scalar_t = float;
        AllReduceDirect<scalar_t> fn;
        auto [bw, valid, seconds] = runbench<scalar_t>(nranks, fn, data_size);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
}
