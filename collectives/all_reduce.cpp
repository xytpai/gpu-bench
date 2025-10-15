#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

template <typename T, int vec_size, int loops>
__global__ void _reduce_kernel(T *dst, const T *src, size_t n) {
    size_t block_work_size = loops * blockDim.x * vec_size;
    size_t index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
#pragma unroll
    for (int i = 0; i < loops; ++i) {
        size_t remaining = n - index;
        if (remaining < vec_size) {
            for (auto ii = index; ii < n; ii++) {
                dst[ii] += src[ii];
            }
        } else {
            using vec_t = aligned_array<T, vec_size>;
            auto src_vec = *reinterpret_cast<vec_t *>(const_cast<T *>(&src[index]));
            auto dst_vec_ptr = reinterpret_cast<vec_t *>(&dst[index]);
            auto dst_vec = *dst_vec_ptr;
#pragma unroll
            for (int ii = 0; ii < vec_size; ++ii) {
                dst_vec.val[ii] += src_vec.val[ii];
            }
            *dst_vec_ptr = dst_vec;
        }
        index += blockDim.x * vec_size;
    }
}

template <typename T, int vec_size, int loops>
void reduce_kernel(T *dst, int dst_dev, const T *src, int src_dev, size_t n, gpuStream_t s) {
    constexpr int block_size = 256;
    constexpr int block_work_size = loops * block_size * vec_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);
    _reduce_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock, 0, s>>>(dst, src, n);
}

template <typename T>
class AllReduceRing {
public:
    AllReduceRing(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        constexpr int vec_size = 16 / sizeof(T);
        int nranks = rs.size();
        size_t chunk_size = rs[0].chunk_size;
        size_t chunk_len = chunk_size / sizeof(T);
        int num_streams = rs[0].num_streams;
        std::vector<int> counters(nranks);
        for (int i = 0; i < nranks; ++i) {
            counters[i] = i;
        }
        for (int ct = 1; ct < nranks; ++ct) {
            for (int rank = 0; rank < nranks; ++rank) {
                int recver = (rank + 1) % nranks;
                int ridx = counters[rank];
                size_t offset = ridx * chunk_size;
                reduce_kernel<T, vec_size, 1>(
                    (T *)(rs[recver].buffers + offset), recver,
                    (T *)(rs[rank].buffers + offset), rank,
                    chunk_len, rs[recver].streams[0]);
                counters[rank] = (ridx + nranks - 1) % nranks;
            }
            for (int r = 0; r < nranks; ++r) {
                gpuSetDevice(r);
                gpuDeviceSynchronize();
            }
        }
        for (int ct = 1; ct < nranks; ++ct) {
            for (int rank = 0; rank < nranks; ++rank) {
                int recver = (rank + 1) % nranks;
                int ridx = counters[rank];
                size_t offset = ridx * chunk_size;
                memcpy_peer_async(
                    rs[recver].buffers + offset, recver,
                    rs[rank].buffers + offset, rank,
                    chunk_size, rs[recver].streams[0],
                    p2p_);
                counters[rank] = (ridx + nranks - 1) % nranks;
            }
            for (int r = 0; r < nranks; ++r) {
                gpuSetDevice(r);
                gpuDeviceSynchronize();
            }
        }
    }

private:
    bool p2p_;
};

template <typename T, typename func_t>
std::tuple<double, bool, double> runbench(int nranks, func_t fn, size_t data_bytes) {
    std::vector<GPUResources> rs;
    assert(data_bytes % nranks == 0);
    size_t chunk_size = data_bytes / nranks;
    allocate_resources(rs, chunk_size, /*nsegments*/ 1, /*nstreams*/ 1);
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
        std::cout << "======== 1GB all reduce ring test ========\n";
        using scalar_t = float;
        AllReduceRing<scalar_t> fn(true);
        auto [bw, valid, seconds] = runbench<scalar_t>(nranks, fn, data_size);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
}
