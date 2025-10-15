#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

void run(int local, std::vector<GPUResources> &rs, bool is_p2p) {
    int nranks = rs.size();
    size_t chunk_size = rs[0].chunk_size;
    size_t segment_size = rs[0].segment_size;
    int num_streams = rs[0].num_streams;
    int c = 0;
    for (size_t idx = 0; idx < chunk_size; idx += segment_size) {
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == local) continue;
            size_t offset_local = local * chunk_size + idx;
            size_t offset_peer = peer * chunk_size + idx;
            size_t remaining = std::min(chunk_size - idx, segment_size);
            // local -> peer
            int stream_peer = (c) % num_streams;
            memcpy_peer_async(rs[peer].buffers + offset_local, peer,
                              rs[local].buffers + offset_local, local,
                              remaining, rs[peer].streams[stream_peer],
                              is_p2p);
            // peer -> local
            int stream_local = (c * nranks + peer) % num_streams;
            memcpy_peer_async(rs[local].buffers + offset_peer, local,
                              rs[peer].buffers + offset_peer, peer,
                              remaining, rs[local].streams[stream_local],
                              is_p2p);
        }
        c++;
    }
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuDeviceSynchronize();
    }
}

std::tuple<double, bool> runbench(int local, size_t buffer_size, size_t segment_size, int nstreams, bool is_p2p) {
    std::vector<GPUResources> rs;
    allocate_resources(rs, buffer_size, segment_size, nstreams);
    int nranks = rs.size();
    for (int w = 0; w < 2; ++w) {
        run(local, rs, is_p2p);
    }
    reset_gather_flags(rs);
    auto t0 = std::chrono::high_resolution_clock::now();
    run(local, rs, is_p2p);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (nranks - 1) * 2 * buffer_size;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    std::vector<std::vector<bool>> mask(nranks);
    for (int local = 0; local < nranks; ++local) {
        mask[local].resize(nranks);
        for (int peer = 0; peer < nranks; ++peer) {
            mask[local][peer] = false;
        }
    }
    for (int i = 0; i < nranks; ++i) {
        mask[i][i] = true;
        mask[local][i] = true;
        mask[i][local] = true;
    }
    bool valid = validate_gather_flags(rs, mask);
    delete_resources(rs);
    return {gbps, valid};
}

int main() {
    int nranks = enable_p2p();
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    size_t segment_size = buffer_size;
    std::cout << "1GB p2p broadcast gather direct test ... \n";
    for (int local = 0; local < nranks; ++local) {
        auto [bw, valid] = runbench(local, buffer_size, segment_size, nranks, true);
        std::cout << "[" << local << "]: " << bw << " GBps --- val:" << valid << "\n";
    }
    std::cout << "1GB uva broadcast gather direct test ... \n";
    for (int local = 0; local < nranks; ++local) {
        auto [bw, valid] = runbench(local, buffer_size, segment_size, nranks, false);
        std::cout << "[" << local << "]: " << bw << " GBps --- val:" << valid << "\n";
    }
}
