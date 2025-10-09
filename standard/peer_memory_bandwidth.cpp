#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

float measure_peer_bandwidth(int dst_dev, int src_dev, size_t bytes) {
    gpuSetDevice(src_dev);
    void *d_src = nullptr;
    gpuMalloc(&d_src, bytes);
    gpuMemset(d_src, 0x7f, bytes);

    gpuSetDevice(dst_dev);
    void *d_dst = nullptr;
    gpuMalloc(&d_dst, bytes);
    gpuMemset(d_dst, 0x00, bytes);

    gpuSetDevice(src_dev);
    gpuStream_t s;
    gpuStreamCreate(&s);

    gpuSetDevice(dst_dev);
    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);

    for (int wm = 0; wm < 2; ++wm) {
        gpuMemcpyPeerAsync(d_dst, dst_dev, d_src, src_dev, bytes, s);
        gpuStreamSynchronize(s);
    }

    gpuSetDevice(dst_dev);
    gpuEventRecord(start);
    gpuMemcpyPeerAsync(d_dst, dst_dev, d_src, src_dev, bytes, s);
    gpuStreamSynchronize(s);
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);

    gpuEventDestroy(start);
    gpuEventDestroy(stop);
    gpuStreamDestroy(s);
    gpuFree(d_src);
    gpuFree(d_dst);

    float total_GBytes = bytes / 1000.0 / 1000.0;
    auto gbps = total_GBytes / ms;

    return gbps;
}

int main() {
    std::cout << "1GB peer copy test ...\n";
    auto gbps = measure_peer_bandwidth(0, 1, 1024 * 1024 * 1024);
    std::cout << "0->1: " << gbps << "GBPS\n";
}
