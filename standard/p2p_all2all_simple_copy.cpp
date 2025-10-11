#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

struct GPUResources {
    size_t buffer_bytes;
    void *send_buffers;
    std::vector<void *> recv_buffers;
    std::vector<gpuStream_t> streams;
};

void allocate_resources(std::vector<GPUResources> &rs, size_t buffer_bytes) {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    rs.resize(ngpus);
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        rs[g].buffer_bytes = buffer_bytes;
        gpuMalloc(&rs[g].send_buffers, buffer_bytes);
        gpuMemset(rs[g].send_buffers, 0xA5, buffer_bytes);
        rs[g].recv_buffers.resize(ngpus);
        for (int rg = 0; rg < ngpus; ++rg) {
            if (rg == g) continue;
            gpuMalloc(&rs[g].recv_buffers[rg], buffer_bytes);
        }
        rs[g].streams.resize(ngpus);
        for (int s = 0; s < ngpus; ++s) {
            gpuStreamCreate(&rs[g].streams[s]);
        }
    }
}

void reset_send_buffers(std::vector<GPUResources> &rs, unsigned char flag) {
    for (int i = 0; i < rs.size(); ++i) {
        gpuSetDevice(i);
        size_t buffer_bytes = rs[i].buffer_bytes;
        gpuMemset(rs[i].send_buffers, (flag + i) % 255, 1);
        gpuMemset((unsigned char *)rs[i].send_buffers + buffer_bytes - 1, (flag + i + 1) % 255, 1);
    }
}

bool validate_recv_buffers(std::vector<GPUResources> &rs, unsigned char flag) {
    auto data = new unsigned char[2];
    size_t buffer_bytes = rs[0].buffer_bytes;
    int ngpus = rs.size();
    bool c0, c1;
    for (int local = 0; local < ngpus; ++local) {
        for (int peer = 0; peer < ngpus; ++peer) {
            if (peer == local) continue;
            gpuSetDevice(local);
            auto ptr = (unsigned char *)rs[local].recv_buffers[peer];
            gpuMemcpy(data, ptr, 1, gpuMemcpyDeviceToHost);
            gpuMemcpy(data + 1, ptr + buffer_bytes - 1, 1, gpuMemcpyDeviceToHost);
            c0 = data[0] == (flag + peer) % 255;
            c1 = data[1] == (flag + peer + 1) % 255;
            if (!(c0 && c1)) return false;
        }
    }
    delete[] data;
    return true;
}

void run_a2a_copy(std::vector<GPUResources> &rs) {
    int ngpus = rs.size();
    for (int local = 0; local < ngpus; ++local) {
        for (int peer = 0; peer < ngpus; ++peer) {
            if (peer == local) continue;
            size_t buffer_bytes = rs[local].buffer_bytes;
            // peer -> local
            gpuSetDevice(local);
            gpuMemcpyPeerAsync(rs[local].recv_buffers[peer], local,
                               rs[peer].send_buffers, peer,
                               buffer_bytes, rs[local].streams[peer]);
        }
    }
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) {
            gpuStreamSynchronize(s);
        }
    }
}

std::tuple<float, bool> measure_p2p_bandwidth(size_t buffer_bytes) {
    // std::cout << "allocate resources ... \n";
    std::vector<GPUResources> rs;
    allocate_resources(rs, buffer_bytes);
    int ngpus = rs.size();

    // std::cout << "warmup ... \n";
    for (int w = 0; w < 2; ++w) {
        run_a2a_copy(rs);
    }

    // std::cout << "run iters ... \n";
    reset_send_buffers(rs, 0xA1);
    auto t0 = std::chrono::high_resolution_clock::now();
    run_a2a_copy(rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (ngpus * ngpus - ngpus) * buffer_bytes;
    float gbps = ((double)nbytes_total / seconds) / 1e9;
    bool valid = validate_recv_buffers(rs, 0xA1);

    // cleanup
    for (int g = 0; g < ngpus; ++g) {
        gpuSetDevice(g);
        for (auto s : rs[g].streams) gpuStreamDestroy(s);
        gpuFree(rs[g].send_buffers);
        for (auto p : rs[g].recv_buffers) gpuFree(p);
    }

    return {gbps, valid};
}

int main() {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    std::cout << "1GB p2p all-to-all simple copy test ... \n";
    size_t buffer_bytes = (size_t)1024 * 1024 * 1024;
    auto [bw, valid] = measure_p2p_bandwidth(buffer_bytes);
    std::cout << "Total: " << bw << " GBps --- val:" << valid << "\n";
    std::cout << "Per GPU: " << bw / ngpus * 2 << " GBps\n";
}
