#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

template <int NRanks, int vec_size = 4>
__global__ void ring_all_gather_kernel(void **workspace, int rank, size_t buffer_size) {
    const int block_work_size = blockDim.x * vec_size;
    int counter = rank;
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    int next_rank = (rank + 1) % NRanks;
    for (int ct = 1; ct < NRanks; ++ct) {
        size_t index_ = blockIdx.x * block_work_size + threadIdx.x * vec_size;
        unsigned char *in = (unsigned char *)comm.comm_bufs[rank] + counter * buffer_size;
        unsigned char *out = (unsigned char *)comm.comm_bufs[next_rank] + counter * buffer_size;
        for (size_t index = index_; index < buffer_size; index += block_work_size * gridDim.x) {
            auto remaining = buffer_size - index;
#ifdef __HIPCC__
            __threadfence();
#endif
            if (remaining < vec_size) {
                for (auto i = index; i < buffer_size; i++) {
                    out[i] = in[i];
                }
            } else {
                using vec_t = aligned_array<unsigned char, vec_size>;
                auto in_vec = reinterpret_cast<vec_t *>(const_cast<unsigned char *>(&in[index]));
                auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
                *out_vec = *in_vec;
            }
        }
        counter = (counter + NRanks - 1) % NRanks;
        barrier.sync();
    }
    comm.update(barrier.m_flag_value);
}

class AllGatherRingBarrier {
public:
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        int chunk_size = rs[0].chunk_size;
        std::vector<GPUWorkSpace> workspaces(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            workspaces[rank].init(rs, rank);
            auto s = rs[rank].streams[0];
            dim3 threadsPerBlock(64);
            dim3 numBlocks(DEFAULT_NCTAS);
            // gpuSetDevice((rank + 1) % nranks);
            switch (nranks) {
            case 8: {
                ring_all_gather_kernel<8><<<numBlocks, threadsPerBlock, 0, s>>>(
                    workspaces[rank].workspace(), rank, chunk_size);
            } break;
            case 4: {
                ring_all_gather_kernel<4><<<numBlocks, threadsPerBlock, 0, s>>>(
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

void ring_worker(int rank, int nranks, HostBarrier &barrier, std::vector<GPUResources> &rs, int stream, bool p2p) {
    int counter = rank;
    int r = stream % rs[rank].next.size();
    int recver = rs[rank].next[r][rank];
    size_t chunk_size = rs[rank].chunk_size;
    size_t num_streams = rs[rank].num_streams;
    size_t segment_size = chunk_size / num_streams;
    for (int ct = 1; ct < nranks; ++ct) {
        size_t offset = counter * chunk_size + stream * segment_size;
        memcpy_peer_async(rs[recver].buffers + offset, recver,
                          rs[rank].buffers + offset, rank,
                          segment_size, rs[recver].streams[stream],
                          p2p);
        counter = rs[rank].prev[r][counter];
        gpuStreamSynchronize(rs[recver].streams[stream]);
        barrier.wait();
    }
}

class AllGatherRingMultiThread {
public:
    AllGatherRingMultiThread(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        int num_streams = rs[0].num_streams;
        HostBarrier barrier(nranks * num_streams);
        std::vector<std::thread> threads;
        for (int rank = 0; rank < nranks; ++rank) {
            for (int s = 0; s < num_streams; ++s) {
                threads.emplace_back(ring_worker, rank, nranks, std::ref(barrier), std::ref(rs), s, p2p_);
            }
        }
        for (auto &t : threads) t.join();
    }

private:
    bool p2p_;
};

void direct_worker(int rank, int nranks, HostBarrier &barrier, std::vector<GPUResources> &rs, bool p2p) {
    size_t chunk_size = rs[rank].chunk_size;
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == rank) continue;
        memcpy_peer_async(rs[rank].buffers + peer * chunk_size, rank,
                          rs[peer].buffers + peer * chunk_size, peer,
                          chunk_size, rs[rank].streams[peer],
                          p2p);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        gpuStreamSynchronize(rs[rank].streams[peer]);
    }
}

class AllGatherDirectMultiThread {
public:
    AllGatherDirectMultiThread(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        int nranks = rs.size();
        HostBarrier barrier(nranks);
        std::vector<std::thread> threads;
        for (int rank = 0; rank < nranks; ++rank) {
            threads.emplace_back(direct_worker, rank, nranks, std::ref(barrier), std::ref(rs), p2p_);
        }
        for (auto &t : threads) t.join();
    }

private:
    bool p2p_;
};

template <typename func_t>
std::tuple<double, bool, double> runbench(func_t fn, size_t buffer_size, size_t segment_size, int nstreams) {
    std::vector<GPUResources> rs;
    allocate_resources(rs, buffer_size, segment_size, nstreams);
    int nranks = rs.size();
    std::vector<std::vector<bool>> mask(nranks);
    for (int local = 0; local < nranks; ++local) {
        mask[local].resize(nranks);
        for (int peer = 0; peer < nranks; ++peer) {
            mask[local][peer] = true;
        }
    }
    for (int w = 0; w < 2; ++w) {
        fn(rs);
    }
    reset_gather_flags(rs);
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    bool valid = validate_gather_flags(rs, mask);
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (nranks - 1) * nranks * buffer_size;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    delete_resources(rs);
    return {gbps, valid, seconds};
}

int main() {
    int nranks = enable_p2p();
    std::cout << "nranks: " << nranks << "\n";
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    {
        std::cout << "======== 1GB p2p all gather ring multi-thread test ========\n";
        AllGatherRingMultiThread fn(true);
        size_t nsegments = 4;
        auto [bw, valid, seconds] = runbench(fn, buffer_size, buffer_size / nsegments, nsegments);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB p2p all gather direct multi-thread test ========\n";
        AllGatherDirectMultiThread fn(true);
        auto [bw, valid, seconds] = runbench(fn, buffer_size, buffer_size, nranks);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB barrier all gather ring test ========\n";
        AllGatherRingBarrier fn;
        auto [bw, valid, seconds] = runbench(fn, buffer_size, 1, 1);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / nranks * 2 << " GBps\n";
    }
}
