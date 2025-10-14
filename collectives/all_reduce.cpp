#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

template <typename T, int NRanks, int vec_size = 4>
__global__ void ring_all_reduce_kernel(void **workspace, int rank, size_t length) {
    const int block_work_size = blockDim.x * vec_size;
    size_t index_ = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    int counter = rank;
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    // reduce scatter
    for (int ct = 1; ct < NRanks; ++ct) {
        T *in = (T *)comm.current_comm_bufs[counter];
        T *out = (T *)comm.next_comm_bufs[counter];
        for (size_t index = index_; index < length; index += block_work_size * gridDim.x) {
            auto remaining = length - index;
            if (remaining < vec_size) {
                for (auto i = index; i < length; i++) {
                    out[i] += in[i];
                }
            } else {
                using vec_t = aligned_array<T, vec_size>;
                auto in_vec = *reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
                auto out_vec_ptr = reinterpret_cast<vec_t *>(&out[index]);
                auto out_vec = *out_vec_ptr;
#pragma unroll
                for (int i = 0; i < vec_size; ++i) {
                    out_vec.val[i] += in_vec.val[i];
                }
                *out_vec_ptr = out_vec;
            }
        }
        counter = (counter + NRanks - 1) % NRanks;
        barrier.sync();
    }
    counter = (counter + 1) % NRanks;
    // all gather
    for (int ct = 1; ct < NRanks; ++ct) {
        T *in = (T *)comm.current_comm_bufs[counter];
        T *out = (T *)comm.next_comm_bufs[counter];
        for (size_t index = index_; index < length; index += block_work_size * gridDim.x) {
            auto remaining = length - index;
            if (remaining < vec_size) {
                for (auto i = index; i < length; i++) {
                    out[i] = in[i];
                }
            } else {
                using vec_t = aligned_array<T, vec_size>;
                auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
                auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
                *out_vec = *in_vec;
            }
        }
        counter = (counter + NRanks - 1) % NRanks;
        barrier.sync();
    }
    comm.update(barrier.m_flag_value);
}

template <typename T>
class AllReduceRing {
public:
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        size_t buffer_size = rs[0].buffer_size;
        size_t length = buffer_size / sizeof(T);
        std::vector<GPUWorkSpace> workspaces(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            workspaces[rank].init(rs, rank);
            auto s = rs[rank].streams[0];
            dim3 threadsPerBlock(256);
            dim3 numBlocks(DEFAULT_NCTAS);
            switch (nranks) {
            case 8: {
                ring_all_reduce_kernel<T, 8><<<numBlocks, threadsPerBlock, 0, s>>>(
                    workspaces[rank].workspace(), rank, length);
            } break;
            case 4: {
                ring_all_reduce_kernel<T, 4><<<numBlocks, threadsPerBlock, 0, s>>>(
                    workspaces[rank].workspace(), rank, length);
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
std::tuple<double, bool, double> runbench(func_t fn, size_t size_bytes) {
    std::vector<GPUResources> rs;
    allocate_resources(rs, size_bytes, size_bytes, 1);
    int ngpus = rs.size();
    for (int w = 0; w < 2; ++w) {
        fn(rs);
    }
    // reset_gather_flags(rs, 0xA3);
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    // bool valid = validate_gather_flags(rs, 0xA3, mask);
    bool valid = false;
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (ngpus - 1) * ngpus * size_bytes;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    delete_resources(rs);
    return {gbps, valid, seconds};
}

int main() {
    int ngpus = enable_p2p();
    std::cout << "ngpus: " << ngpus << "\n";
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    {
        std::cout << "======== 1GB barrier all reduce ring test ========\n";
        using scalar_t = float;
        AllReduceRing<scalar_t> fn;
        auto [bw, valid, seconds] = runbench<scalar_t>(fn, buffer_size);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
}
