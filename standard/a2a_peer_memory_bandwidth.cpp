#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "utils.h"
using namespace std;

struct GPUResources {
    std::vector<void *> buffers;
    std::vector<gpuStream_t> streams;
};

struct Path {
    int local;
    int peer;
};

std::vector<Path> enable_p2p(int *ngpus) {
    gpuGetDeviceCount(ngpus);
    std::vector<Path> paths;
    for (int local = 0; local < *ngpus; ++local) {
        gpuSetDevice(local);
        for (int peer = 0; peer < *ngpus; ++peer) {
            if (local == peer) continue;
            int can = 0;
            gpuDeviceCanAccessPeer(&can, local, peer);
            if (can) {
                gpuDeviceEnablePeerAccess(peer, 0);
                paths.push_back({local, peer});
            }
        }
    }
    return paths;
}

void measure_peer_bandwidth(size_t buffer_bytes, int num_buffers, int streams_per_gpu, int iters, int warmups = 2) {
    int ngpus = 0;
    auto paths = enable_p2p(&ngpus);

    // allocate resources
    std::vector<GPUResources> rs(ngpus);
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        rs[g].buffers.resize(num_buffers);
        for (int b = 0; b < num_buffers; ++b) {
            gpuMalloc(&rs[g].buffers[b], buffer_bytes);
            // Initialize memory with something to avoid zero-page optimizations
            gpuMemset(rs[g].buffers[b], 0xA5, 1);
        }
        rs[g].streams.resize(streams_per_gpu);
        for (int s = 0; s < streams_per_gpu; ++s) {
            gpuStreamCreate(&rs[g].streams[s]);
        }
    }

    // warmup
    for (int w = 0; w < warmups; ++w) {
        for (auto &p : paths) {
            int local = p.local;
            int peer = p.peer;
            for (int b = 0; b < num_buffers; ++b) {
                int s_peer = b % streams_per_gpu;
                gpuMemcpyPeerAsync(rs[peer].buffers[b], peer, rs[local].buffers[b], local, buffer_bytes, rs[peer].streams[s_peer]);
            }
        }
        for (int g = 0; g < ngpus; ++g) {
            gpuSetDevice(g);
            gpuDeviceSynchronize();
        }
    }

    // run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        for (auto &p : paths) {
            int local = p.local;
            int peer = p.peer;
            for (int b = 0; b < num_buffers; ++b) {
                int s_peer = b % streams_per_gpu;
                gpuMemcpyPeerAsync(rs[peer].buffers[b], peer, rs[local].buffers[b], local, buffer_bytes, rs[peer].streams[s_peer]);
            }
        }
        for (int g = 0; g < ngpus; ++g) {
            gpuSetDevice(g);
            gpuDeviceSynchronize();
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    size_t path_count = paths.size();
    size_t nbytes_total = path_count * num_buffers * buffer_bytes * iters;
    double bw_total = ((double)nbytes_total / seconds) / 1e9;

    std::cout << "num paths: " << paths.size() << "\n";
    std::cout << "num buffers per GPU: " << num_buffers << "\n";
    std::cout << "buffer bytes: " << buffer_bytes / 1024 / 1024 << " MB\n";
    std::cout << "bandwidth total: " << bw_total << " GBps\n";

    // cleanup
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) gpuStreamDestroy(s);
        for (auto b : rs[g].buffers) gpuFree(b);
    }
}

int main() {
    std::cout << "a2a copy test ... \n";
    int nbuffers = 32;
    int buffer_bytes = 1024 * 1024 * 1024 / 32;
    measure_peer_bandwidth(buffer_bytes, nbuffers, nbuffers, 1);
}
