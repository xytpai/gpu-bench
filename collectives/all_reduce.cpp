#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

template <int NRanks, typename T, int vec_size = 2>
__global__ void all_reduce_kernel(void **workspace, int rank, size_t chunk_size) {
    const int block_work_size = blockDim.x * vec_size;
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    size_t index_ = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    size_t chunk_len = chunk_size / sizeof(T);
    size_t offset = rank * chunk_len;
    for (size_t index = index_; index < chunk_len; index += block_work_size * gridDim.x) {
        auto remaining = chunk_len - index;
        if (remaining < vec_size) {
            // for (int index_v = index; index_v < chunk_len; ++index_v) {
            //     size_t offset_element = (index_v + offset) * sizeof(T);
            //     auto acc = (T)0;
            //     for (int peer = 0; peer < NRanks; ++peer) {
            //         acc += *reinterpret_cast<T *>((char *)comm.comm_bufs[peer] + offset_element);
            //     }
            //     for (int peer = 0; peer < NRanks; ++peer) {
            //         *reinterpret_cast<T *>((char *)comm.comm_bufs[peer] + offset_element) = acc;
            //     }
            // }
        } else {
            using vec_t = aligned_array<T, vec_size>;
            vec_t vec_out;
#pragma unroll
            for (int v = 0; v < vec_size; ++v) {
                vec_out.val[v] = 0;
            }
            size_t offset_v = (index + offset) * sizeof(T);
            for (int peer = 0; peer < NRanks; ++peer) {
                auto vec_in = *reinterpret_cast<vec_t *>((char *)comm.comm_bufs[peer] + offset_v);
#pragma unroll
                for (int v = 0; v < vec_size; ++v) {
                    vec_out.val[v] += vec_in.val[v];
                }
            }
            for (int peer = 0; peer < NRanks; ++peer) {
                *reinterpret_cast<vec_t *>((char *)comm.comm_bufs[peer] + offset_v) = vec_out;
            }
        }
    }
    comm.update(barrier.m_flag_value);
}

template <typename T>
class AllReduceDirect {
public:
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        int chunk_size = rs[0].chunk_size;
        std::vector<GPUWorkSpace> workspaces(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            workspaces[rank].init(rs, rank);
        }
        for (int rank = 0; rank < nranks; ++rank) {
            gpuSetDevice(rank);
            dim3 threadsPerBlock(256);
            dim3 numBlocks(DEFAULT_NCTAS);
            switch (nranks) {
            case 8: {
                all_reduce_kernel<8, T><<<numBlocks, threadsPerBlock>>>(
                    workspaces[rank].workspace(), rank, chunk_size);
            } break;
            case 4: {
                all_reduce_kernel<4, T><<<numBlocks, threadsPerBlock>>>(
                    workspaces[rank].workspace(), rank, chunk_size);
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
    // assert(data_bytes % nranks == 0);
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
    size_t data_size = (size_t)1024 * 1024 * 1024 + 4;
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
