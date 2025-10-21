#include <iostream>
#include <random>

#include "utils.h"
using namespace std;
namespace cg = cooperative_groups;

template <int NRanks>
__global__ void all_gpu_barrier_test_kernel(void **workspace) {
    SyncComm<NRanks> comm(workspace);
    int rank = comm.rank;
    Barrier<NRanks> barrier(rank, comm);
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < 4; ++i) {
        if (threadIdx.x == 0) {
            printf("This is rank %d block %d from loop %d\n", rank, blockIdx.x, i);
        }
        barrier.sync();
        grid.sync();
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
        void **ptr = workspaces[rank].workspace();
        void *args[] = {(void *)&ptr};
        dim3 numBlocks(8);
        switch (nranks) {
        case 8: {
            gpuLaunchCooperativeKernel(all_gpu_barrier_test_kernel<8>, numBlocks, threadsPerBlock, args, 0, nullptr);
        } break;
        case 4: {
            gpuLaunchCooperativeKernel(all_gpu_barrier_test_kernel<4>, numBlocks, threadsPerBlock, args, 0, nullptr);
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
    enable_p2p();
    std::cout << "all gpu barrier test ... \n";
    all_gpu_barrier_test();
    std::cout << "ok\n";
}
