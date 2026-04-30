// filename: matmul.cu
// purpose: Implements naive CUDA matrix multiplication kernel and launcher.
// phase: Phase 2 - Real GPU
// last modified: 2026-04-29

#include "matmul.cuh"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace opengpu::backends::cuda::kernels {

#if defined(__CUDACC__)
namespace {

constexpr std::size_t kThreadsPerBlockX = 16U;
constexpr std::size_t kThreadsPerBlockY = 16U;

/**
 * @brief CUDA kernel for naive square matrix multiplication.
 * @param a_device Device pointer for matrix A.
 * @param b_device Device pointer for matrix B.
 * @param c_device Device pointer for matrix C.
 * @param n Matrix dimension.
 * @return None.
 * @sideeffects Writes output values to device memory.
 */
__global__ void matmul_naive_kernel(const float* a_device, const float* b_device,
                                    float* c_device, const std::size_t n) {
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  const std::size_t col =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

  if (row >= n || col >= n) {
    return;
  }

  float accum = 0.0F;
  for (std::size_t k = 0; k < n; ++k) {
    accum += a_device[(row * n) + k] * b_device[(k * n) + col];
  }
  c_device[(row * n) + col] = accum;
}

}  // namespace
#endif

bool launch_naive_matmul(const float* a_device, const float* b_device, float* c_device,
                         const std::size_t n) {
#if defined(__CUDACC__)
  if (a_device == nullptr || b_device == nullptr || c_device == nullptr || n == 0U) {
    return false;
  }

  const dim3 threads_per_block(static_cast<unsigned int>(kThreadsPerBlockX),
                               static_cast<unsigned int>(kThreadsPerBlockY), 1U);
  const unsigned int grid_x = static_cast<unsigned int>(
      (n + kThreadsPerBlockX - 1U) / kThreadsPerBlockX);
  const unsigned int grid_y = static_cast<unsigned int>(
      (n + kThreadsPerBlockY - 1U) / kThreadsPerBlockY);
  const dim3 blocks_per_grid(grid_x, grid_y, 1U);

  matmul_naive_kernel<<<blocks_per_grid, threads_per_block>>>(a_device, b_device,
                                                              c_device, n);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return false;
  }
  return true;
#else
  (void)a_device;
  (void)b_device;
  (void)c_device;
  (void)n;
  // TODO(phase-2): Build with CUDA compiler to execute kernel launch.
  return false;
#endif
}

}  // namespace opengpu::backends::cuda::kernels
// test
