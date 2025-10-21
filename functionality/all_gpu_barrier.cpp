#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

template <int NRanks>
__global__ void all_gpu_barrier_test_kernel(void **workspace, int rank) {
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    for (int i = 0; i < 4; ++i) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("This is rank %d from loop %d\n", rank, i);
        }
        barrier.sync();
    }
    comm.update(barrier.m_flag_value);
}

void all_gpu_barrier_test() {
    std::vector<GPUResources> rs;
    size_t chunk_size = 1024;
    auto nranks = allocate_resources(rs, chunk_size, chunk_size, 1);
    std::vector<GPUWorkSpace> workspaces(nranks);
    for (int rank = 0; rank < nranks; ++rank) {
        workspaces[rank].init(rs, rank);
    }
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        dim3 threadsPerBlock(256);
        dim3 numBlocks(256);
        switch (nranks) {
        case 8: {
            all_gpu_barrier_test_kernel<8><<<numBlocks, threadsPerBlock>>>(
                workspaces[rank].workspace(), rank);
        } break;
        case 4: {
            all_gpu_barrier_test_kernel<4><<<numBlocks, threadsPerBlock>>>(
                workspaces[rank].workspace(), rank);
        } break;
        default:
            return;
        }
    }
    for (int g = 0; g < nranks; ++g) {
        gpuSetDevice(g);
        gpuDeviceSynchronize();
    }
}

int main() {
    std::cout << "all gpu barrier test ... \n";
    all_gpu_barrier_test();
    std::cout << "ok\n";
}
