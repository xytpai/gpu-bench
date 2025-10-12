#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

void sync_devices(int dst, int src) {
    gpuSetDevice(src);
    gpuDeviceSynchronize();
    gpuSetDevice(dst);
    gpuDeviceSynchronize();
}

float measure_peer_bandwidth(int dst_dev, int src_dev, size_t bytes) {
    gpuSetDevice(src_dev);
    void *d_src = nullptr;
    gpuMalloc(&d_src, bytes);
    gpuMemset(d_src, 0x7f, bytes);
    gpuDeviceSynchronize();

    gpuSetDevice(dst_dev);
    void *d_dst = nullptr;
    gpuMalloc(&d_dst, bytes);
    gpuMemset(d_dst, 0x00, bytes);
    gpuDeviceSynchronize();

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);

    for (int wm = 0; wm < 2; ++wm) {
        gpuMemcpyPeerAsync(d_dst, dst_dev, d_src, src_dev, bytes);
        sync_devices(dst_dev, src_dev);
    }

    gpuSetDevice(dst_dev);
    gpuEventRecord(start);
    gpuMemcpyPeerAsync(d_dst, dst_dev, d_src, src_dev, bytes);
    sync_devices(dst_dev, src_dev);
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);

    gpuEventDestroy(start);
    gpuEventDestroy(stop);
    gpuFree(d_src);
    gpuFree(d_dst);

    float total_GBytes = bytes / 1000.0 / 1000.0;
    auto gbps = total_GBytes / ms;

    return gbps;
}

int main() {
    std::cout << "1GB peer single dir copy test ... (GBps)\n";
    enable_p2p();
    int device_count = 0;
    gpuGetDeviceCount(&device_count);
    std::cout << std::right << std::setw(11) << "dst/src";
    for (int j = 0; j < device_count; ++j) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
    }
    std::cout << "\n";
    for (int dst = 0; dst < device_count; ++dst) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(dst) + "]");
        for (int src = 0; src < device_count; ++src) {
            auto bw = measure_peer_bandwidth(dst, src, (size_t)1 * 1024 * 1024 * 1024);
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
        }
        std::cout << "\n";
    }
}
