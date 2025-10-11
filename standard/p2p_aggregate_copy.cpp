#include <iostream>
#include <random>
#include <vector>

#include "utils.h"
using namespace std;

struct GPUResources {
    size_t buffer_bytes;
    void *send_buffers;
    std::vector<void *> recv_buffers;
    std::vector<gpuStream_t> streams;
};

void allocate_resources(int local, std::vector<GPUResources> &rs, size_t buffer_bytes) {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    rs.resize(ngpus);
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        rs[g].buffer_bytes = buffer_bytes;
        gpuMalloc(&rs[g].send_buffers, buffer_bytes);
        // Initialize memory with something to avoid zero-page optimizations
        gpuMemset(rs[g].send_buffers, 0xA5, 1);
        rs[g].recv_buffers.resize(ngpus);
        if (g == local) {
            for (int rg = 0; rg < ngpus; ++rg) {
                if (rg == g) continue;
                gpuMalloc(&rs[g].recv_buffers[rg], buffer_bytes);
            }
        } else {
            gpuMalloc(&rs[g].recv_buffers[local], buffer_bytes);
        }
        rs[g].streams.resize(ngpus);
        for (int s = 0; s < ngpus; ++s) {
            gpuStreamCreate(&rs[g].streams[s]);
        }
    }
}

void run_aggregate_copy(int local, std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int peer = 0; peer < ngpus; ++peer) {
        if (peer == local) continue;
        size_t buffer_bytes = rs[local].buffer_bytes;
        // local -> peer
        gpuSetDevice(peer);
        gpuMemcpyPeerAsync(rs[peer].recv_buffers[local], peer,
                           rs[local].send_buffers, local,
                           buffer_bytes, rs[peer].streams[local]);
        // peer -> local
        gpuSetDevice(local);
        gpuMemcpyPeerAsync(rs[local].recv_buffers[peer], local,
                           rs[peer].send_buffers, peer,
                           buffer_bytes, rs[local].streams[peer]);
    }
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        gpuDeviceSynchronize();
    }
}

float measure_p2p_bandwidth(int local, size_t buffer_bytes) {
    // std::cout << "allocate resources ... \n";
    std::vector<GPUResources> rs;
    allocate_resources(local, rs, buffer_bytes);
    int ngpus = rs.size();

    // std::cout << "warmup ... \n";
    for (int w = 0; w < 2; ++w) {
        run_aggregate_copy(local, rs);
    }

    // std::cout << "run iters ... \n";
    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);
    run_aggregate_copy(local, rs);
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);
    double seconds = ms / 1000.0;
    size_t nbytes_total = (ngpus - 1) * 2 * buffer_bytes;
    float gbps = ((double)nbytes_total / seconds) / 1e9;

    // cleanup
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) gpuStreamDestroy(s);
        gpuFree(rs[g].send_buffers);
        for (auto p : rs[g].recv_buffers) gpuFree(p);
    }

    return gbps;
}

int main() {
    std::cout << "1GB p2p aggregate copy test ... (GBps)\n";
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    size_t buffer_bytes = (size_t)1024 * 1024 * 1024;
    for (int local = 0; local < ngpus; ++local) {
        auto bw = measure_p2p_bandwidth(local, buffer_bytes);
        std::cout << "[" << local << "]: " << bw << " GBps\n";
    }
}
