#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;

void sync_devices(int dst, int src) {
    gpuSetDevice(src);
    gpuDeviceSynchronize();
    gpuSetDevice(dst);
    gpuDeviceSynchronize();
}

class SingleCopy {
public:
    SingleCopy(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(int src, int dst, std::vector<GPUResources> &rs) {
        size_t chunk_size = rs[0].chunk_size;
        size_t segment_size = rs[0].segment_size;
        int num_streams = rs[0].num_streams;
        int c = 0;
        for (size_t idx = 0; idx < chunk_size; idx += segment_size) {
            int stream = (c++) % num_streams;
            size_t remaining = std::min(chunk_size - idx, segment_size);
            // src -> dst
            memcpy_peer_async(
                rs[dst].buffers + src * chunk_size + idx, dst,
                rs[src].buffers + src * chunk_size + idx, src,
                remaining, rs[dst].streams[stream],
                p2p_);
        }
        sync_devices(dst, src);
    }

private:
    bool p2p_;
};

class BiCopy {
public:
    BiCopy(bool p2p) :
        p2p_(p2p) {
    }
    void operator()(int dev0, int dev1, std::vector<GPUResources> &rs) {
        size_t chunk_size = rs[0].chunk_size;
        size_t segment_size = rs[0].segment_size;
        int num_streams = rs[0].num_streams;
        int c = 0;
        for (size_t idx = 0; idx < chunk_size; idx += segment_size) {
            int stream = (c++) % num_streams;
            size_t remaining = std::min(chunk_size - idx, segment_size);
            size_t offset0 = dev0 * chunk_size + idx;
            size_t offset1 = dev1 * chunk_size + idx;
            // dev0 -> dev1
            memcpy_peer_async(
                rs[dev1].buffers + offset0, dev1,
                rs[dev0].buffers + offset0, dev0,
                remaining, rs[dev1].streams[stream],
                p2p_);
            // dev1 -> dev0
            memcpy_peer_async(
                rs[dev0].buffers + offset1, dev0,
                rs[dev1].buffers + offset1, dev1,
                remaining, rs[dev0].streams[stream],
                p2p_);
        }
        sync_devices(dev0, dev1);
    }

private:
    bool p2p_;
};

template <typename func_t>
std::tuple<double, bool> runbench(func_t fn, int src, int dst, size_t buffer_size, size_t segment_size, int nstreams, bool bidir) {
    std::vector<GPUResources> rs;
    allocate_resources(rs, buffer_size, segment_size, nstreams);
    int nranks = rs.size();
    for (int w = 0; w < 2; ++w) {
        fn(src, dst, rs);
    }
    reset_gather_flags(rs);
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(src, dst, rs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = buffer_size;
    if (bidir) nbytes_total *= 2;
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    std::vector<std::vector<bool>> mask(nranks);
    for (int local = 0; local < nranks; ++local) {
        mask[local].resize(nranks);
        for (int peer = 0; peer < nranks; ++peer) {
            mask[local][peer] = false;
        }
    }
    mask[dst][src] = true;
    if (bidir) mask[src][dst] = true;
    bool valid = validate_gather_flags(rs, mask);
    delete_resources(rs);
    return {gbps, valid};
}

int main() {
    int device_count = enable_p2p();
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    int nstreams = 1;
    size_t segment_size = buffer_size / nstreams;
    {
        SingleCopy fn(true);
        std::cout << "======== 1GB p2p single dir copy test (GBps) ========\n";
        std::cout << std::right << std::setw(11) << "src/dst";
        for (int j = 0; j < device_count; ++j) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
        }
        std::cout << "\n";
        for (int src = 0; src < device_count; ++src) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(src) + "]");
            for (int dst = 0; dst < device_count; ++dst) {
                auto [bw, valid] = runbench(fn, src, dst, buffer_size, segment_size, nstreams, false);
                assert(valid);
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
            }
            std::cout << "\n";
        }
    }
    {
        SingleCopy fn(false);
        std::cout << "======== 1GB uva single dir copy test (GBps) ========\n";
        std::cout << std::right << std::setw(11) << "src/dst";
        for (int j = 0; j < device_count; ++j) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
        }
        std::cout << "\n";
        for (int src = 0; src < device_count; ++src) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(src) + "]");
            for (int dst = 0; dst < device_count; ++dst) {
                auto [bw, valid] = runbench(fn, src, dst, buffer_size, segment_size, nstreams, false);
                assert(valid);
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
            }
            std::cout << "\n";
        }
    }
    {
        BiCopy fn(true);
        std::cout << "======== 1GB p2p bi dir copy test (GBps) ========\n";
        std::cout << std::right << std::setw(11) << "bi-dir";
        for (int j = 0; j < device_count; ++j) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
        }
        std::cout << "\n";
        for (int src = 0; src < device_count; ++src) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(src) + "]");
            for (int dst = 0; dst < device_count; ++dst) {
                double bw = 0;
                if (src < dst) {
                    auto r = runbench(fn, src, dst, buffer_size, segment_size, nstreams, true);
                    bw = std::get<0>(r);
                    auto valid = std::get<1>(r);
                    assert(valid);
                }
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
            }
            std::cout << "\n";
        }
    }
    {
        BiCopy fn(false);
        std::cout << "======== 1GB uva bi dir copy test (GBps) ========\n";
        std::cout << std::right << std::setw(11) << "bi-dir";
        for (int j = 0; j < device_count; ++j) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(j) + "]");
        }
        std::cout << "\n";
        for (int src = 0; src < device_count; ++src) {
            std::cout << std::right << std::setw(11) << ("[" + std::to_string(src) + "]");
            for (int dst = 0; dst < device_count; ++dst) {
                double bw = 0;
                if (src < dst) {
                    auto r = runbench(fn, src, dst, buffer_size, segment_size, nstreams, true);
                    bw = std::get<0>(r);
                    auto valid = std::get<1>(r);
                    assert(valid);
                }
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << bw << " ";
            }
            std::cout << "\n";
        }
    }
}
