// filename: test_profiler.cpp
// purpose: Validates profiler report and optimization insight generation.
// phase: Phase 7 - Profiling
// last modified: 2026-04-29

#include "backends/cuda/cuda_backend.h"
#include "backends/rtl_sim/rtl_sim_backend.h"
#include "matmul.h"
#include "profiler/profiler.h"
#include "runtime/kernel.h"
#include "runtime/launch_config.h"
#include "scheduler/scheduler.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kDim = 64U;
constexpr std::size_t kElemCount = kDim * kDim;
constexpr std::size_t kOpsPerMatmul = 2U * kDim * kDim * kDim;
constexpr float kTolerance = 1.0e-5F;

/**
 * @brief Builds deterministic matrix values for repeatable tests.
 * @param n Matrix dimension.
 * @param seed Seed offset for pattern generation.
 * @return Matrix data in row-major order.
 * @sideeffects None.
 */
std::vector<float> make_matrix(const std::size_t n, const std::size_t seed) {
  std::vector<float> out(n * n, 0.0F);
  constexpr std::size_t kValueMod = 23U;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<float>((i + seed) % kValueMod);
  }
  return out;
}

/**
 * @brief Computes max absolute difference between vectors.
 * @param lhs First vector.
 * @param rhs Second vector.
 * @return Maximum absolute element-wise difference.
 * @sideeffects None.
 */
float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(lhs[i] - rhs[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

/**
 * @brief Converts elapsed time to throughput in ops/s.
 * @param latency_ms Elapsed latency in milliseconds.
 * @return Throughput in operations per second.
 * @sideeffects None.
 */
double throughput_from_latency(const double latency_ms) {
  if (latency_ms <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(kOpsPerMatmul) / (latency_ms / 1000.0);
}

}  // namespace

int main() {
  const std::vector<float> a = make_matrix(kDim, 3U);
  const std::vector<float> b = make_matrix(kDim, 11U);

  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{48U, 1U, 1U},
  };
  opengpu::scheduler::WarpScheduler warp_scheduler(launch);
  const opengpu::scheduler::SchedulerMetrics sched = warp_scheduler.simulate();

  const auto cpu_begin = std::chrono::steady_clock::now();
  const std::vector<float> cpu_out = opengpu::backends::cpu::kernels::matmul(a, b, kDim);
  const auto cpu_end = std::chrono::steady_clock::now();
  const double cpu_latency_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

  opengpu::backends::cuda::CUDABackend cuda_backend;
  const auto cuda_begin = std::chrono::steady_clock::now();
  std::vector<float> cuda_out;
  const bool cuda_ok = cuda_backend.run_matmul(a, b, kDim, &cuda_out);
  const auto cuda_end = std::chrono::steady_clock::now();
  const double cuda_latency_ms =
      std::chrono::duration<double, std::milli>(cuda_end - cuda_begin).count();
  if (!cuda_ok || cuda_out.size() != kElemCount) {
    std::cerr << "CUDA backend matmul failed\n";
    return EXIT_FAILURE;
  }

  opengpu::backends::rtl_sim::RTLSimBackend rtl_backend;
  rtl_backend.set_pending_matmul_inputs(a, b, kDim);
  const opengpu::runtime::Kernel kernel{"matmul", 1U};
  const auto rtl_begin = std::chrono::steady_clock::now();
  const bool rtl_ok = rtl_backend.launch(kernel, launch);
  const auto rtl_end = std::chrono::steady_clock::now();
  const double rtl_latency_ms =
      std::chrono::duration<double, std::milli>(rtl_end - rtl_begin).count();
  const std::vector<float> rtl_out = rtl_backend.last_output();
  if (!rtl_ok || rtl_out.size() != kElemCount) {
    std::cerr << "RTL sim backend matmul failed\n";
    return EXIT_FAILURE;
  }

  if (max_abs_diff(cpu_out, cuda_out) > kTolerance || max_abs_diff(cpu_out, rtl_out) > kTolerance) {
    std::cerr << "Backend outputs diverged\n";
    return EXIT_FAILURE;
  }

  opengpu::profiler::Profiler profiler;
  profiler.record(opengpu::profiler::Metrics{
      "cpu", cpu_latency_ms, throughput_from_latency(cpu_latency_ms), sched.occupancy,
      sched.stall_fraction, sched.memory_coalesced, sched.low_occupancy, sched.high_stall});
  profiler.record(opengpu::profiler::Metrics{
      "cuda", cuda_latency_ms, throughput_from_latency(cuda_latency_ms), sched.occupancy,
      sched.stall_fraction, sched.memory_coalesced, sched.low_occupancy, sched.high_stall});
  profiler.record(opengpu::profiler::Metrics{
      "rtl_sim", rtl_latency_ms, throughput_from_latency(rtl_latency_ms), sched.occupancy,
      sched.stall_fraction, sched.memory_coalesced, sched.low_occupancy, sched.high_stall});

  std::cout << std::fixed << std::setprecision(3);
  profiler.report(true);
  if (!sched.low_occupancy && sched.memory_coalesced && !sched.high_stall) {
    std::cerr << "Expected at least one scheduler-driven insight trigger\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
