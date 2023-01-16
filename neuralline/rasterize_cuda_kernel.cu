#include <ATen/ATen.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#ifndef MAX_THREADS
#define MAX_THREADS 512
#endif


#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


#define GPU_ERROR_CHECK(ans) {gpu_assert((ans), __FILE__, __LINE__);}
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"\nGPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}


inline int gpu_blocks(int total_threads, int threads_per_block) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}


namespace {

template <typename scalar_t=float>
__device__ __forceinline__ scalar_t vec2f_squared_norm(const scalar_t* v) {
    return v[0] * v[0] + v[1] * v[1];
}


template <typename scalar_t=float>
__global__ void rasterize_cuda_forward_kernel(
    scalar_t*       __restrict__ line_map,
    int32_t*        __restrict__ line_index_map,
    scalar_t*       __restrict__ line_weight_map,
    int32_t*        __restrict__ locks,
    int                          num_intensity_channels,
    int                          num_lines,
    int                          loops,
    const scalar_t* __restrict__ lines,
    const scalar_t* __restrict__ intensities,
    int                          img_size,
    scalar_t                     thickness,
    scalar_t                     eps) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const int batchId = i / num_lines;
    const int lineId  = i % num_lines;
    const scalar_t* line = &lines[i * 3];
    if (line[2] > 0 || lineId == (num_lines - 1))
        return;

    scalar_t tl[2][2]; 
    for (int vidx = 0; vidx < 2; ++vidx)
        for (int axis = 0; axis < 2; ++axis) 
            tl[vidx][axis] = (line[vidx * 3 + axis] + 1.0f) * (img_size - 1.0f) / 2.0f;
    
    scalar_t tlVec[2]      = {tl[1][0] - tl[0][0], tl[1][1] - tl[0][1]};
    const scalar_t length2 = vec2f_squared_norm<scalar_t>(tlVec);
    const scalar_t length  = sqrt(length2);
    if (length < eps) return;

    int xiMin = max(floor(min(tl[0][0], tl[1][0]) - thickness), 0.0f);
    int xiMax = min(ceil(max(tl[0][0], tl[1][0]) + thickness), img_size - 1.0f);
    int yiMin = max(floor(min(tl[0][1], tl[1][1]) - thickness), 0.0f);
    int yiMax = min(ceil(max(tl[0][1], tl[1][1]) + thickness), img_size - 1.0f);
    
    const int imgSize2 = img_size * img_size;

    for (int xi = xiMin; xi <= xiMax; ++xi) {
        for (int yi = yiMin; yi <= yiMax; ++yi) {
            scalar_t pv0Vec[2] = {xi - tl[0][0], yi - tl[0][1]};
            scalar_t ratio     = max(min((pv0Vec[0] * tlVec[0] + pv0Vec[1] * tlVec[1]) / length2, 1.0f), 0.0f);
            scalar_t pProj[2]  = {tl[0][0] + ratio * tlVec[0], tl[0][1] + ratio * tlVec[1]};
            scalar_t ppProj[2] = {xi - pProj[0], yi - pProj[1]};
            scalar_t dist      = sqrt(vec2f_squared_norm<scalar_t>(ppProj));

            if (dist > thickness) continue;
            
            int lockId = imgSize2 * batchId + img_size * yi + xi;
            int locked = 0;
            do {
                if ((locked = atomicCAS(&locks[lockId], 0, 1)) == 0) {
                    if (atomicAdd(&line_index_map[lockId], 0) < lineId) {
                        atomicExch(&line_index_map[lockId], lineId);
                        atomicExch(&line_weight_map[lockId], ratio);
                        const scalar_t* intensity = &intensities[i * num_intensity_channels];
                        for (int cid = 0; cid < num_intensity_channels; ++cid) {
                            scalar_t lerp_val = intensity[cid] + ratio * (intensity[num_intensity_channels + cid] - intensity[cid]);
                            int pixId = num_intensity_channels * imgSize2 * batchId + imgSize2 * cid + img_size * yi + xi;
                            atomicExch(&line_map[pixId], lerp_val);
                        }
                    }
                    atomicExch(&locks[lockId], 0);
                }
            } while (locked > 0);
        }
    }
}


template <typename scalar_t=float>
__global__ void rasterize_cuda_backward_kernel_intensities(
    scalar_t*       __restrict__   grad_intensities,
    const int32_t*  __restrict__   line_index_map,
    const scalar_t* __restrict__   line_weight_map,
    int                            num_intensity_channels,
    int                            num_lines,
    int                            batch_size,
    int                            loops,
    const scalar_t* __restrict__   grad_line_map,
    int                            img_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const int localLineId = line_index_map[i];
    if (localLineId < 0)
        return;
    const scalar_t ratio = line_weight_map[i];
    
    const int imgSize2 = img_size * img_size;
    const int batchId = i / imgSize2;
    const int pixId = i % imgSize2;
    const int globalLineId = num_lines * batchId + localLineId;
    
    for (int cid = 0; cid < num_intensity_channels; ++cid) {
        const scalar_t gradLine = grad_line_map[num_intensity_channels * imgSize2 * batchId + imgSize2 * cid + pixId];
        atomicAdd(&grad_intensities[globalLineId * num_intensity_channels + cid], (1 - ratio) * gradLine);
        atomicAdd(&grad_intensities[(globalLineId + 1) * num_intensity_channels + cid], ratio * gradLine);
    }
}

} 


void rasterize_cuda_forward(
    at::Tensor lines,
    at::Tensor intensities,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor locks,
    int   img_size,
    float thickness,
    float eps) {

    const auto batch_size = lines.size(0);
    const auto num_lines = lines.size(1);
    const auto num_intensity_channels = intensities.size(2);
 
    const int loops   = batch_size * num_lines;
    const int threads = MAX_THREADS;
    const int blocks  = gpu_blocks(loops, threads);

    rasterize_cuda_forward_kernel<float><<<blocks, threads>>>(
        line_map.data<float>(),
        line_index_map.data<int32_t>(),
        line_weight_map.data<float>(),
        locks.data<int32_t>(),
        num_intensity_channels,
        num_lines,
        loops,
        lines.data<float>(),
        intensities.data<float>(),
        img_size,
        thickness,
        eps);
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}


void rasterize_cuda_backward(
    at::Tensor grad_intensities,
    at::Tensor grad_line_map,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor lines,
    at::Tensor intensities,
    int   img_size,
    float thickness,
    float eps) {

    const auto batch_size = lines.size(0);
    const auto num_lines = lines.size(1);
    const auto num_intensity_channels = intensities.size(2);
    
    const int loops  = batch_size * img_size * img_size;
    const int threads = MAX_THREADS;
    const int blocks = gpu_blocks(loops, threads);

    rasterize_cuda_backward_kernel_intensities<float><<<blocks, threads>>>(
        grad_intensities.data<float>(),
        line_index_map.data<int32_t>(),
        line_weight_map.data<float>(),
        num_intensity_channels,
        num_lines,
        batch_size,
        loops,
        grad_line_map.data<float>(),
        img_size);
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
} 
