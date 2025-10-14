#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

void run(int local, std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    size_t chunk_size = rs[0].chunk_size;
    int num_chunks = rs[0].num_chunks;
    int num_streams = rs[0].num_streams;
    for (int c = 0; c < num_chunks; ++c) {
        for (int peer = 0; peer < ngpus; ++peer) {
            if (peer == local) continue;
            // local -> peer
            int stream_peer = c % num_streams;
            memcpy_peer_async(rs[peer].buffers[local][c], peer,
                              rs[local].buffers[local][c], local,
                              chunk_size, rs[peer].streams[stream_peer],
                              true);
            // peer -> local
            int stream_local = (c * ngpus + peer) % num_streams;
            memcpy_peer_async(rs[local].buffers[peer][c], local,
                              rs[peer].buffers[peer][c], peer,
                              chunk_size, rs[local].streams[stream_local],
                              true);
        }
    }
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        gpuDeviceSynchronize();
    }
}

std::tuple<double, bool> runbench(int local, size_t buffer_size, size_t chunk_size, int nstreams) {
    std::vector<GPUResources> rs;
    allocate_gather_resources(rs, buffer_size, chunk_size, nstreams);
    int ngpus = rs.size();
    for (int w = 0; w < 2; ++w) {
        run(local, rs);
    }
    reset_gather_flags(rs, 0xA1);
    auto t0 = std::chrono::high_resolution_clock::now();
    run(local, rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (ngpus - 1) * 2 * buffer_size;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    std::vector<std::vector<bool>> mask(ngpus);
    for (int local = 0; local < ngpus; ++local) {
        mask[local].resize(ngpus);
        for (int peer = 0; peer < ngpus; ++peer) {
            mask[local][peer] = false;
        }
    }
    for (int i = 0; i < ngpus; ++i) {
        mask[i][i] = true;
        mask[local][i] = true;
        mask[i][local] = true;
    }
    bool valid = validate_gather_flags(rs, 0xA1, mask);
    delete_gather_resources(rs);
    return {gbps, valid};
}

int main() {
    std::cout << "1GB p2p broadcast gather direct test ... \n";
    int ngpus = enable_p2p();
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    size_t chunk_size = buffer_size;
    for (int local = 0; local < ngpus; ++local) {
        auto [bw, valid] = runbench(local, buffer_size, chunk_size, ngpus);
        std::cout << "[" << local << "]: " << bw << " GBps --- val:" << valid << "\n";
    }
}
