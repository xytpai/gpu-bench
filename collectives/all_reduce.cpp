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
    gpuSetDevice(dst_dev);
    constexpr int block_size = 256;
    constexpr int block_work_size = loops * block_size * vec_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);
    _reduce_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock, 0, s>>>(dst, src, n);
}

template <typename T>
void ring_worker(int rank, int nranks, HostBarrier &barrier, std::vector<GPUResources> &rs, bool p2p) {
    constexpr int vec_size = 16 / sizeof(T);
    int num_streams = rs[0].num_streams;
    int counter = rank;
    int recver = (rank + 1) % nranks;
    int sender = (rank + nranks - 1) % nranks;
    size_t chunk_size = rs[rank].chunk_size;
    int copy_stream_id = 0;
    int compute_stream_sid = 1;
    int produce_idx = 0;
    int consume_idx = 1;
    for (int ct = 1; ct < nranks; ++ct) {
        int sidx = (counter + nranks - 1) % nranks;
        memcpy_peer_async(
            rs[rank].buffers + (nranks + produce_idx) * chunk_size, rank,
            rs[sender].buffers + sidx * chunk_size, sender,
            chunk_size, rs[rank].streams[copy_stream_id],
            p2p);
        produce_idx ^= 1;
        consume_idx ^= 1;
        gpuStreamSynchronize(rs[rank].streams[copy_stream_id]);
        gpuStreamSynchronize(rs[rank].streams[compute_stream_sid]);
        barrier.wait();
        reduce_kernel<T, vec_size, 1>(
            (T *)(rs[rank].buffers + sidx * chunk_size), rank,
            (T *)(rs[rank].buffers + (nranks + consume_idx) * chunk_size), rank,
            chunk_size / sizeof(T), rs[rank].streams[copy_stream_id]);
        counter = sidx;
    }
    gpuStreamSynchronize(rs[rank].streams[compute_stream_sid]);
    barrier.wait();
#ifdef __HIPCC__
    int c = 0;
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == (rank + 1) % nranks) continue;
        int s = (c++) % num_streams;
        int peer_cid = (peer + 1) % nranks;
        memcpy_peer_async(rs[rank].buffers + peer * chunk_size, rank,
                          rs[peer].buffers + peer_cid * chunk_size, peer,
                          chunk_size, rs[rank].streams[s],
                          p2p);
    }
#else
    for (int ct = 1; ct < nranks; ++ct) {
        int ridx = counter;
        size_t offset = ridx * chunk_size;
        memcpy_peer_async(
            rs[recver].buffers + offset, recver,
            rs[rank].buffers + offset, rank,
            chunk_size, rs[recver].streams[copy_stream_id],
            p2p);
        counter = (ridx + nranks - 1) % nranks;
        gpuStreamSynchronize(rs[recver].streams[copy_stream_id]);
        barrier.wait();
    }
#endif
    for (int s = 0; s < num_streams; ++s) {
        gpuStreamSynchronize(rs[rank].streams[s]);
    }
}

template <typename T>
class AllReduceRingMultiThread {
public:
    AllReduceRingMultiThread(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        HostBarrier barrier(nranks);
        std::vector<std::thread> threads;
        for (int rank = 0; rank < nranks; ++rank) {
            threads.emplace_back(ring_worker<T>, rank, nranks, std::ref(barrier), std::ref(rs), p2p_);
        }
        for (auto &t : threads) t.join();
    }

private:
    bool p2p_;
};

template <typename T, typename func_t>
std::tuple<double, bool, double> runbench(int nranks, func_t fn, size_t data_bytes) {
    std::vector<GPUResources> rs;
    assert(data_bytes % nranks == 0);
    size_t chunk_size = data_bytes / nranks;
    allocate_resources(rs, chunk_size,
                       /*segment_size*/ chunk_size,
                       /*nstreams*/ nranks,
                       /*alloc size*/ (nranks + 2) * chunk_size);
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
        AllReduceRingMultiThread<scalar_t> fn(true);
        auto [bw, valid, seconds] = runbench<scalar_t>(nranks, fn, data_size);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
}
