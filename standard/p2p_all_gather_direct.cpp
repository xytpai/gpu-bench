#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

struct GPUResources {
    size_t buffer_bytes;
    std::vector<void *> buffers;
    std::vector<gpuStream_t> streams;
};

void allocate_resources(std::vector<GPUResources> &rs, size_t buffer_bytes) {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    rs.resize(ngpus);
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        rs[g].buffer_bytes = buffer_bytes;
        rs[g].buffers.resize(ngpus);
        for (int rg = 0; rg < ngpus; ++rg) {
            gpuMalloc(&rs[g].buffers[rg], buffer_bytes);
        }
        rs[g].streams.resize(ngpus);
        for (int s = 0; s < ngpus; ++s) {
            gpuStreamCreate(&rs[g].streams[s]);
        }
    }
}

void reset_buffers(std::vector<GPUResources> &rs, unsigned char flag) {
    for (int i = 0; i < rs.size(); ++i) {
        gpuSetDevice(i);
        size_t buffer_bytes = rs[i].buffer_bytes;
        gpuMemset(rs[i].buffers[i], (flag + i) % 255, 1);
        gpuMemset((unsigned char *)rs[i].buffers[i] + buffer_bytes - 1, (flag + i + 1) % 255, 1);
        gpuDeviceSynchronize();
    }
}

bool validate_buffers(std::vector<GPUResources> &rs, unsigned char flag) {
    auto data = new unsigned char[2];
    size_t buffer_bytes = rs[0].buffer_bytes;
    int ngpus = rs.size();
    bool c0, c1;
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        for (int peer = 0; peer < ngpus; ++peer) {
            auto ptr = (unsigned char *)rs[local].buffers[peer];
            gpuMemcpy(data, ptr, 1, gpuMemcpyDeviceToHost);
            gpuMemcpy(data + 1, ptr + buffer_bytes - 1, 1, gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            c0 = data[0] == (flag + peer) % 255;
            c1 = data[1] == (flag + peer + 1) % 255;
            if (!(c0 && c1)) return false;
        }
    }
    delete[] data;
    return true;
}

void run_p2p(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int local = 0; local < ngpus; ++local) {
        for (int peer = 0; peer < ngpus; ++peer) {
            if (peer == local) continue;
            size_t buffer_bytes = rs[local].buffer_bytes;
            // peer -> local
            gpuSetDevice(local);
            gpuMemcpyPeerAsync(rs[local].buffers[peer], local,
                               rs[peer].buffers[peer], peer,
                               buffer_bytes, rs[local].streams[peer]);
        }
    }
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        gpuDeviceSynchronize();
    }
}

std::tuple<float, bool, double> measure_p2p_bandwidth(size_t buffer_bytes) {
    // std::cout << "allocate resources ... \n";
    std::vector<GPUResources> rs;
    allocate_resources(rs, buffer_bytes);
    int ngpus = rs.size();

    // std::cout << "warmup ... \n";
    for (int w = 0; w < 2; ++w) {
        run_p2p(rs);
    }

    // std::cout << "run iters ... \n";
    reset_buffers(rs, 0xA3);
    auto t0 = std::chrono::high_resolution_clock::now();
    run_p2p(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (ngpus - 1) * ngpus * buffer_bytes;
    float gbps = ((double)nbytes_total / seconds) / 1e9;
    bool valid = validate_buffers(rs, 0xA3);

    // cleanup
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) gpuStreamDestroy(s);
        for (auto p : rs[g].buffers) gpuFree(p);
    }

    return {gbps, valid, seconds};
}

int main() {
    std::cout << "1GB p2p all gather direct test ... \n";
    enable_p2p();
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    size_t buffer_bytes = (size_t)1024 * 1024 * 1024;
    auto [bw, valid, seconds] = measure_p2p_bandwidth(buffer_bytes);
    std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
    std::cout << "Latency: " << seconds * 1000000 << " us\n";
    std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
}
