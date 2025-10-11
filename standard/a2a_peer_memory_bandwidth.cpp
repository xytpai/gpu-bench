#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "utils.h"
using namespace std;

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

struct GPUResources {
    int buffer_bytes;
    std::vector<void *> send_buffers;
    std::vector<void *> recv_buffers;
    std::vector<gpuStream_t> streams;
};

void run_a2a(std::vector<Path> &paths, std::vector<GPUResources> &rs) {
    for (auto &p : paths) {
        int local = p.local;
        int peer = p.peer;
        gpuSetDevice(peer);
        int nbuffers = rs[peer].recv_buffers.size();
        int nstreams = rs[peer].streams.size();
        int buffer_bytes = rs[peer].buffer_bytes;
        // local -> peer
        for (int b = 0; b < nbuffers; ++b) {
            int s_peer = b % nstreams;
            gpuMemcpyPeerAsync(rs[peer].recv_buffers[b], peer, rs[local].send_buffers[b],
                               local, buffer_bytes, rs[peer].streams[s_peer]);
        }
    }
    for (int g = 0; g < rs.size(); ++g) {
        gpuSetDevice(g);
        gpuDeviceSynchronize();
    }
}

void validate(std::vector<GPUResources> &rs, unsigned char val) {
    for (int g = 0; g < rs.size(); ++g) {
        gpuSetDevice(g);
        int bb = rs[g].buffer_bytes;
        for (auto &ptr_ : rs[g].recv_buffers) {
            auto ptr = (unsigned char *)ptr_;
            bool c0 = val == *ptr;
            bool c1 = val == *(ptr + bb - 1);
            if (c0 && c1) {
                // do nothing;
            } else {
                std::cout << "val:error!\n";
                return;
            }
        }
    }
}

void measure_peer_bandwidth(size_t buffer_bytes, int num_buffers, int streams_per_gpu, int iters, int warmups = 2) {
    int ngpus = 0;
    auto paths = enable_p2p(&ngpus);

    // allocate resources
    std::vector<GPUResources> rs(ngpus);
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        rs[g].send_buffers.resize(num_buffers);
        rs[g].recv_buffers.resize(num_buffers);
        rs[g].buffer_bytes = buffer_bytes;
        for (int b = 0; b < num_buffers; ++b) {
            gpuMalloc(&rs[g].send_buffers[b], buffer_bytes);
            gpuMalloc(&rs[g].recv_buffers[b], buffer_bytes);
            // Initialize memory with something to avoid zero-page optimizations
            gpuMemset(rs[g].send_buffers[b], 0xA5, buffer_bytes);
            gpuMemset(rs[g].recv_buffers[b], 0x01, buffer_bytes);
        }
        rs[g].streams.resize(streams_per_gpu);
        for (int s = 0; s < streams_per_gpu; ++s) {
            gpuStreamCreate(&rs[g].streams[s]);
        }
    }

    // warmup
    for (int w = 0; w < warmups; ++w) {
        run_a2a(paths, rs);
        validate(rs, 0xA5);
    }

    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (int b = 0; b < num_buffers; ++b) {
            gpuMemset(rs[g].send_buffers[b], 0xA6, buffer_bytes);
            gpuMemset(rs[g].recv_buffers[b], 0x02, buffer_bytes);
        }
    }

    // run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_a2a(paths, rs);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    size_t path_count = paths.size();
    size_t nbytes_total = path_count * num_buffers * buffer_bytes * iters;
    double bw_total = ((double)nbytes_total / seconds) / 1e9;

    std::cout << "num paths: " << paths.size() << "\n";
    std::cout << "num buffers per gpu: " << num_buffers << "\n";
    std::cout << "buffer bytes: " << buffer_bytes / 1024 / 1024 << " MB\n";
    std::cout << "streams per gpu: " << streams_per_gpu << "\n";
    std::cout << "bandwidth total: " << bw_total << " GBps\n";

    validate(rs, 0xA6);

    // cleanup
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) gpuStreamDestroy(s);
        for (auto b : rs[g].send_buffers) gpuFree(b);
        for (auto b : rs[g].recv_buffers) gpuFree(b);
    }
}

int main() {
    std::cout << "a2a copy test ... \n";
    size_t buffer_bytes = (size_t)1024 * 1024 * 1024 / 8;
    int num_buffers = 128;
    int streams_per_gpu = 64;
    measure_peer_bandwidth(buffer_bytes, num_buffers, streams_per_gpu, 1);
}
