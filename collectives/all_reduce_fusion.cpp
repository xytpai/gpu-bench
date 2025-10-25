#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;
namespace cg = cooperative_groups;

#define NBLOCKS_PER_GPU 256

namespace allreduce_fusion {

namespace details {

static constexpr int kBytesPerAccess = 16;

} // namespace details

namespace block_utils {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];
    const int tid = threadIdx.x;
    const int w_tid = tid % 32;
    const int wid = tid / 32;
    val = warp_reduce_sum(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < blockDim.x / 32; ++i) {
            shared[0] += shared[i];
        }
    }
    __syncthreads();
    return shared[0];
}

} // namespace block_utils

namespace comm {

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (*reinterpret_cast<int volatile *>(counter_ptr) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void **workspace, int rank) {
        counter_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[2];
        clear_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[4];
        flag_value = *flag_ptr;
        int comm_size = reinterpret_cast<int *>(workspace[NRanks * 3])[3];
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + r]) + static_cast<int64_t>(data_offset) * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (*reinterpret_cast<int volatile *>(counter_ptr) != gridDim.x) {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    int *clear_ptr;
    uint8_t *data_bufs[NRanks];
    uint8_t *clear_buf;
    int clear_size;
    int flag_value;
};

template <int NRanks>
class Barrier {
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const &comm) {
        if (threadIdx.x < NRanks) {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int *>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag = reinterpret_cast<int *>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        constexpr int kBarrierFlagCount = DEFAULT_NCTAS;
        __syncthreads();
        __threadfence_system();
        if (threadIdx.x < NRanks) {
            m_flag_value = next_flag(m_flag_value);
            // To avoid the ABA problem, we need to synchronize the correct flag value to all
            // barrier_flags, even if the corresponding CTA has not been launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x) {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
            }
        }
        __syncthreads();
    }

protected:
    __device__ __forceinline__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        __hip_atomic_store(addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
    }

    __device__ __forceinline__ int ld_flag(int *addr) {
        int flag;
#ifdef __CUDACC__
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                     : "=r"(flag)
                     : "l"(addr));
#else
        flag = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag) {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag) {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};

} // namespace comm

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t {
    T data[vec_size];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(T *ptr) {
        *this = *reinterpret_cast<vec_t<T, vec_size> *>(ptr);
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, vec_size> *>(ptr) = *this;
    }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE> &self,
                                         const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[i] = (float)self[i] + (float)other[i];
    }
}

template <typename T>
struct AllReduceFusionParams {
    int nranks;
    int rank;
    int size;
    int hidden_dim;
    void **workspace;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    float rms_eps;
};

template <typename T>
class FusedOp {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

public:
    __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const &params, int access_id,
                                       int access_id_in_token) :
        m_params(params),
        m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
        m_gamma_val.load(reinterpret_cast<T *>(params.rms_gamma) + m_access_id_in_token * VEC_SIZE);
        m_residual_val.load(reinterpret_cast<T *>(params.residual_in) + m_access_id * VEC_SIZE);
    }

    __device__ __forceinline__ void update(int access_id) {
        if (m_access_id != access_id) {
            m_access_id = access_id;
            m_residual_val.load(reinterpret_cast<T *>(m_params.residual_in) + m_access_id * VEC_SIZE);
        }
    }

    __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {
        // val.store(reinterpret_cast<T *>(m_params.allreduce_out) + m_access_id * VEC_SIZE);
        vec_add_<T, VEC_SIZE>(val, m_residual_val);
        val.store(reinterpret_cast<T *>(m_params.residual_out) + m_access_id * VEC_SIZE);
        val = rms_norm(val, m_gamma_val);
        val.store(reinterpret_cast<T *>(m_params.norm_out) + m_access_id * VEC_SIZE);
    }

protected:
    __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const &residual,
                                                           vec_t<T, VEC_SIZE> const &gamma) {
        __shared__ float s_val;
        vec_t<T, VEC_SIZE> norm_out;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float v = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]);
            acc += v * v;
        }
        acc = block_utils::block_reduce_sum<float>(acc);
        if (threadIdx.x == 0) {
            s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            reinterpret_cast<T *>(&norm_out)[i] =
                static_cast<T>(static_cast<float>(reinterpret_cast<T const *>(&residual)[i]) * s_val * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]));
        }
        return norm_out;
    }

private:
    AllReduceFusionParams<T> const &m_params;
    int m_access_id;
    int m_access_id_in_token;
    vec_t<T, VEC_SIZE> m_residual_val;
    vec_t<T, VEC_SIZE> m_gamma_val;
};

template <typename T, int NRanks, bool Fp32Acc>
__global__ void allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params,
                                                     std::array<int, NRanks> begin_tokens,
                                                     std::array<int, NRanks> token_num_per_ranks) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_id = blockIdx.x;
    int access_id_in_token = threadIdx.x;
    int token_stride = gridDim.x;
    int access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    int access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    int tot_access = params.size / VEC_SIZE;

    FusedOp<T> fused_op(params, access_id, access_id_in_token);
    comm::SyncComm<NRanks> comm(params.workspace);

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
        int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
        for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
            reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx] =
                reinterpret_cast<float4 *>(params.allreduce_in)[idx];
        }
    }

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

    int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
        vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx * VEC_SIZE);
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
            vec_add_<T, VEC_SIZE>(vals[0], vals[r]);
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[0].store(reinterpret_cast<T *>(comm.comm_bufs[r]) + (tot_access + idx) * VEC_SIZE);
        }
    }

    barrier.sync();

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
        int comm_token_id = token_id + begin_tokens[r];
        int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
        for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
             idx += access_stride, tidx += token_stride) {
            fused_op.update(idx);
            vec_t<T, VEC_SIZE> sum_val;
            sum_val.load(reinterpret_cast<T *>(comm.comm_bufs[params.rank]) + (tot_access + idx) * VEC_SIZE);
            fused_op(sum_val, tidx);
        }
    }

    comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks, bool Fp32Acc = false>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    std::array<int, NRanks> begin_tokens, token_num_per_ranks;
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    for (int r = 0; r < NRanks; ++r) {
        begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
        token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
        if (params.rank == 0)
            std::cout << "rank:" << r << ", begin_tokens:" << begin_tokens[r] << ", token_num_per_ranks:" << token_num_per_ranks[r] << "\n";
    }
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    int threads_per_block = threads_per_token;
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocks(NBLOCKS_PER_GPU);
    if (params.rank == 0)
        std::cout << "threadsPerBlock:" << threadsPerBlock.x << ", numBlocks:" << numBlocks.x << "\n";
    allreduce_fusion_kernel_twoshot_sync<T, NRanks, Fp32Acc><<<numBlocks, threadsPerBlock>>>(params, begin_tokens, token_num_per_ranks);
}

} // namespace allreduce_fusion

namespace test {

struct GPUResources {
    int size;
    int hidden_dim;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    void *comm_bufs;
    // barrier
    int nblocks;
    int *barrier_flags;
    int *counter;
    int *flag;
    // lamport
    int *lamport_flag;
    int *lamport_clear;
    int *lamport_comm_size;
};

class GPUWorkSpace {
public:
    GPUWorkSpace() :
        workspace_(nullptr) {
    }
    void init(std::vector<GPUResources> &rs, int rank) {
        gpuSetDevice(rank);
        int nranks = rs.size();
        auto &r = rs[rank];
        gpuMemset(r.barrier_flags, 0, r.nblocks * nranks * sizeof(int));
        gpuMemset(r.counter, 0, sizeof(int));
        gpuMemset(r.flag, 0, sizeof(int));
        std::vector<void *> workspace(nranks * 3 + 2);
        for (int peer = 0; peer < nranks; ++peer) {
            workspace[peer] = (void *)rs[peer].comm_bufs;
            workspace[nranks + peer] = (void *)rs[peer].barrier_flags;
        }
        workspace[nranks * 3 + 0] = (void *)r.counter;
        workspace[nranks * 3 + 1] = (void *)r.flag;
        gpuMalloc(&workspace_, workspace.size() * sizeof(void *));
        gpuMemcpy(workspace_, workspace.data(), workspace.size() * sizeof(void *), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }
    ~GPUWorkSpace() {
        gpuFree(workspace_);
    }
    void **workspace() const {
        return workspace_;
    }

private:
    void **workspace_;
};

template <typename T>
int allocate_resources(std::vector<GPUResources> &rs, int size, int hidden_dim) {
    int nranks = 0;
    gpuGetDeviceCount(&nranks);
    rs.resize(nranks);
    int num_tokens = size / hidden_dim;
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        rs[rank].size = size;
        rs[rank].hidden_dim = hidden_dim;
        gpuMalloc(&rs[rank].allreduce_in, size * sizeof(T));
        gpuMalloc(&rs[rank].residual_in, size * sizeof(T));
        gpuMalloc(&rs[rank].residual_out, size * sizeof(T));
        gpuMalloc(&rs[rank].norm_out, size * sizeof(T));
        gpuMalloc(&rs[rank].rms_gamma, hidden_dim * sizeof(T));
        gpuMalloc(&rs[rank].comm_bufs, 2 * size * sizeof(T));
        // barrier
        gpuMalloc(&rs[rank].barrier_flags, NBLOCKS_PER_GPU * nranks * sizeof(int));
        gpuMalloc(&rs[rank].counter, sizeof(int));
        gpuMalloc(&rs[rank].flag, sizeof(int));
        rs[rank].nblocks = NBLOCKS_PER_GPU;
    }
    return nranks;
}

void delete_resources(std::vector<GPUResources> &rs) {
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuFree(rs[rank].allreduce_in);
        gpuFree(rs[rank].residual_in);
        gpuFree(rs[rank].residual_out);
        gpuFree(rs[rank].norm_out);
        gpuFree(rs[rank].rms_gamma);
        gpuFree(rs[rank].comm_bufs);
        // barrier
        gpuFree(rs[rank].barrier_flags);
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
    }
}

void allreduce_rmsnorm_ref(
    const float *allreduce_in,
    const float *residual_in,
    const float *rms_gamma,
    int size,
    int hidden_dim,
    int nranks,
    float *residual_out,
    float *norm_out,
    float eps = 1e-6) {
    auto allreduce_out = new float[size];
    // get rank 0
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] = allreduce_in[i];
    }
    // reduce all ranks
    for (int r = 1; r < nranks; ++r) {
        for (int i = 0; i < size; ++i) {
            allreduce_out[i] += allreduce_in[r * size + i];
        }
    }
    // residual
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] += residual_in[i];
        residual_out[i] = allreduce_out[i];
    }
    // norm
    int num_tokens = size / hidden_dim;
    for (int t = 0; t < num_tokens; ++t) {
        double x2 = 0;
        int offset_token = t * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h) {
            auto data = allreduce_out[offset_token + h];
            x2 += data * data;
        }
        double beta = (double)1.0 / std::sqrt(x2 / hidden_dim + eps);
        for (int h = 0; h < hidden_dim; ++h) {
            norm_out[offset_token + h] = allreduce_out[offset_token + h] * beta;
            norm_out[offset_token + h] *= rms_gamma[h];
        }
    }
    delete[] allreduce_out;
}

void runbench(int nranks, int size, int hidden_dim, float eps = 1e-6, float atol = 0.1) {
    // input
    auto allreduce_in = new float[nranks * size];
    auto residual_in = new float[size];
    auto rms_gamma = new float[hidden_dim];

    // output
    auto residual_out_ref = new float[size];
    auto norm_out_ref = new float[size];
    auto residual_out = new float[size];
    auto norm_out = new float[size];

    // gen data
    for (int i = 0; i < nranks * size; ++i) {
        allreduce_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < size; ++i) {
        residual_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < hidden_dim; ++i) {
        rms_gamma[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }

    std::vector<GPUResources> rs;

    // gen gpu data
    allocate_resources<float>(rs, size, hidden_dim);
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(rs[r].allreduce_in, allreduce_in + r * size, size * sizeof(float), gpuMemcpyHostToDevice);
        gpuMemcpy(rs[r].residual_in, residual_in, size * sizeof(float), gpuMemcpyHostToDevice);
        gpuMemcpy(rs[r].rms_gamma, rms_gamma, hidden_dim * sizeof(float), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    std::vector<GPUWorkSpace> workspaces(nranks);
    for (int rank = 0; rank < nranks; ++rank) {
        workspaces[rank].init(rs, rank);
    }

    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        allreduce_fusion::AllReduceFusionParams<float> params;
        params.nranks = nranks;
        params.rank = rank;
        params.size = size;
        params.hidden_dim = hidden_dim;
        params.workspace = workspaces[rank].workspace();
        params.allreduce_in = rs[rank].allreduce_in;
        params.residual_in = rs[rank].residual_in;
        params.residual_out = rs[rank].residual_out;
        params.norm_out = rs[rank].norm_out;
        params.rms_gamma = rs[rank].rms_gamma;
        params.rms_eps = eps;
        if (nranks == 8) {
            allreduce_fusion::allreduce_fusion_kernel_launcher<float, 8>(params);
        } else if (nranks == 4) {
            allreduce_fusion::allreduce_fusion_kernel_launcher<float, 4>(params);
        }
    }
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuDeviceSynchronize();
    }

    allreduce_rmsnorm_ref(
        allreduce_in, residual_in, rms_gamma,
        size, hidden_dim, nranks, residual_out_ref, norm_out_ref, eps);

    bool val = true;
    int print_count = 0;
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(residual_out, rs[r].residual_out, size * sizeof(float), gpuMemcpyDeviceToHost);
        gpuMemcpy(norm_out, rs[r].norm_out, size * sizeof(float), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        for (int i = 0; i < size; ++i) {
            if (std::abs(residual_out[i] - residual_out_ref[i]) > atol) {
                std::cout << "residual_out:" << residual_out[i] << ", residual_out_ref:" << residual_out_ref[i] << "\n";
                if (++print_count == 100) break;
                val = false;
            }
            if (std::abs(norm_out[i] - norm_out_ref[i]) > atol) {
                std::cout << "norm_out:" << norm_out[i] << ", norm_out_ref:" << norm_out_ref[i] << "\n";
                if (++print_count == 100) break;
                val = false;
            }
        }
    }
    std::cout << "validation:" << val << "\n";

    delete[] allreduce_in;
    delete[] residual_in;
    delete[] rms_gamma;
    delete[] residual_out_ref;
    delete[] norm_out_ref;
    delete[] residual_out;
    delete[] norm_out;
    delete_resources(rs);
}

} // namespace test

int main() {
    int nranks = enable_p2p();
    constexpr int num_tokens = 256;
    constexpr int hidden_dim = 1024;
    constexpr int size = num_tokens * hidden_dim;
    test::runbench(nranks, size, hidden_dim);
    std::cout << "ok\n";
}
