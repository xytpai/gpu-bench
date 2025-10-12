#include <iostream>
#include <random>
#include <chrono>

#include "utils.h"
using namespace std;

void sync_devices(int dst, int src) {
    gpuSetDevice(src);
    gpuDeviceSynchronize();
    gpuSetDevice(dst);
    gpuDeviceSynchronize();
}

float measure_peer_bandwidth(int dev0, int dev1, size_t bytes) {
    void *send0, *recv0, *send1, *recv1;

    gpuSetDevice(dev0);
    gpuMalloc(&send0, bytes);
    gpuMalloc(&recv0, bytes);
    gpuMemset(send0, 0x7f, bytes);
    gpuDeviceSynchronize();

    gpuSetDevice(dev1);
    gpuMalloc(&send1, bytes);
    gpuMalloc(&recv1, bytes);
    gpuMemset(send1, 0x6f, bytes);
    gpuDeviceSynchronize();

    for (int wm = 0; wm < 2; ++wm) {
        gpuSetDevice(dev0);
        gpuMemcpyPeerAsync(recv0, dev0, send1, dev1, bytes);
        gpuSetDevice(dev1);
        gpuMemcpyPeerAsync(recv1, dev1, send0, dev0, bytes);
        sync_devices(dev0, dev1);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    gpuSetDevice(dev0);
    gpuMemcpyPeerAsync(recv0, dev0, send1, dev1, bytes);
    gpuSetDevice(dev1);
    gpuMemcpyPeerAsync(recv1, dev1, send0, dev0, bytes);
    sync_devices(dev0, dev1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = 2 * bytes;
    float gbps = ((double)nbytes_total / seconds) / 1e9;

    gpuFree(send0);
    gpuFree(recv0);
    gpuFree(send1);
    gpuFree(recv1);

    return gbps;
}

int main() {
    std::cout << "1GB peer full dir copy test ... (GBps)\n";
    enable_p2p();
    int device_count = 0;
    gpuGetDeviceCount(&device_count);
    std::cout << std::right << std::setw(11) << "bi-dir";
    for (int j = 0; j < device_count; ++j) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
    }
    std::cout << "\n";
    for (int dst = 0; dst < device_count; ++dst) {
        std::cout << std::right << std::setw(11) << ("[" + std::to_string(dst) + "]");
        for (int src = 0; src < device_count; ++src) {
            float bw = 0;
            if (dst < src) {
                bw = measure_peer_bandwidth(dst, src, (size_t)1 * 1024 * 1024 * 1024);
            }
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
        }
        std::cout << "\n";
    }
}
