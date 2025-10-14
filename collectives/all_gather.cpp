#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

class AllGatherDirect {
public:
    AllGatherDirect(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        int ngpus = rs.size();
        for (int local = 0; local < ngpus; ++local) {
            int num_chunks = rs[local].num_chunks;
            int num_streams = rs[local].num_streams;
            size_t chunk_size = rs[local].chunk_size;
            // peer -> local
            for (int c = 0; c < num_chunks; ++c) {
                for (int peer = 0; peer < ngpus; ++peer) {
                    if (peer == local) continue;
                    int s = (c * num_chunks + peer) % num_streams;
                    memcpy_peer_async(rs[local].buffers[peer][c], local,
                                      rs[peer].buffers[peer][c], peer,
                                      chunk_size, rs[local].streams[s],
                                      p2p_);
                }
            }
        }
        for (int g = 0; g < ngpus; ++g) {
            gpuSetDevice(g);
            gpuDeviceSynchronize();
        }
    }

private:
    bool p2p_;
};

class AllGatherRing {
public:
    AllGatherRing(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(std::vector<GPUResources> &rs) {
        int ngpus = rs.size();
        size_t chunk_size = rs[0].chunk_size;
        int num_chunks = rs[0].num_chunks;
        int num_streams = rs[0].num_streams;
        std::vector<std::vector<int>> counters(ngpus);
        for (int i = 0; i < ngpus; ++i) {
            counters[i].resize(num_chunks);
            for (int c = 0; c < num_chunks; ++c) {
                counters[i][c] = i;
            }
        }
        for (int ct = 1; ct < ngpus; ++ct) {
            for (int c = 0; c < num_chunks; ++c) {
                int s = c % num_streams;
                for (int local = 0; local < ngpus; ++local) {
                    int recver = (local + 1) % ngpus;
                    int ridx = counters[local][c];
                    memcpy_peer_async(rs[recver].buffers[ridx][c], recver,
                                      rs[local].buffers[ridx][c], local,
                                      chunk_size, rs[recver].streams[s],
                                      p2p_);
                    counters[local][c] = (ridx + ngpus - 1) % ngpus;
                }
            }
            for (int g = 0; g < ngpus; ++g) {
                gpuSetDevice(g);
                gpuDeviceSynchronize();
            }
        }
    }

private:
    bool p2p_;
};

template <int NRanks, int vec_size = 4>
__global__ void ring_all_gather_kernel(void **workspace, int rank, size_t buffer_size) {
    const int block_work_size = blockDim.x * vec_size;
    int counter = rank;
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    for (int ct = 1; ct < NRanks; ++ct) {
        size_t index_ = blockIdx.x * block_work_size + threadIdx.x * vec_size;
        unsigned char *in = (unsigned char *)comm.current_comm_bufs[counter];
        unsigned char *out = (unsigned char *)comm.next_comm_bufs[counter];
        for (size_t index = index_; index < buffer_size; index += block_work_size * gridDim.x) {
            auto remaining = buffer_size - index;
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
        int buffer_size = rs[0].buffer_size;
        std::vector<GPUWorkSpace> workspaces(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            workspaces[rank].init(rs, rank);
            auto s = rs[rank].streams[0];
            dim3 threadsPerBlock(256);
            dim3 numBlocks(DEFAULT_NCTAS);
            switch (nranks) {
            case 8: {
                ring_all_gather_kernel<8><<<numBlocks, threadsPerBlock, 0, s>>>(
                    workspaces[rank].workspace(), rank, buffer_size);
            } break;
            case 4: {
                ring_all_gather_kernel<4><<<numBlocks, threadsPerBlock, 0, s>>>(
                    workspaces[rank].workspace(), rank, buffer_size);
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

template <typename func_t>
std::tuple<double, bool, double> runbench(func_t fn, size_t buffer_size, size_t chunk_size, int nstreams) {
    std::vector<GPUResources> rs;
    allocate_gather_resources(rs, buffer_size, chunk_size, nstreams);
    int ngpus = rs.size();
    std::vector<std::vector<bool>> mask(ngpus);
    for (int local = 0; local < ngpus; ++local) {
        mask[local].resize(ngpus);
        for (int peer = 0; peer < ngpus; ++peer) {
            mask[local][peer] = true;
        }
    }
    for (int w = 0; w < 2; ++w) {
        fn(rs);
    }
    reset_gather_flags(rs, 0xA3);
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    bool valid = validate_gather_flags(rs, 0xA3, mask);
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (ngpus - 1) * ngpus * buffer_size;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    delete_gather_resources(rs);
    return {gbps, valid, seconds};
}

int main() {
    int ngpus = enable_p2p();
    std::cout << "ngpus: " << ngpus << "\n";
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    size_t nchunks_ring = 4;
    {
        std::cout << "======== 1GB p2p all gather direct test ========\n";
        size_t chunk_size = (size_t)1024 * 1024 * 1024;
        AllGatherDirect fn(true);
        auto [bw, valid, seconds] = runbench(fn, buffer_size, chunk_size, ngpus);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB p2p all gather ring test ========\n";
        size_t chunk_size = (size_t)1024 * 1024 * 1024 / nchunks_ring;
        AllGatherRing fn(true);
        auto [bw, valid, seconds] = runbench(fn, buffer_size, chunk_size, nchunks_ring);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB uva all gather direct test ========\n";
        size_t chunk_size = (size_t)1024 * 1024 * 1024;
        AllGatherDirect fn(false);
        auto [bw, valid, seconds] = runbench(fn, buffer_size, chunk_size, ngpus);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB uva all gather ring test ========\n";
        size_t chunk_size = (size_t)1024 * 1024 * 1024 / nchunks_ring;
        AllGatherRing fn(false);
        auto [bw, valid, seconds] = runbench(fn, buffer_size, chunk_size, nchunks_ring);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
    {
        std::cout << "======== 1GB barrier all gather ring test ========\n";
        AllGatherRingBarrier fn;
        auto [bw, valid, seconds] = runbench(fn, buffer_size, buffer_size, 1);
        std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
        std::cout << "Latency: " << seconds * 1000000 << " us\n";
        std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
    }
}
