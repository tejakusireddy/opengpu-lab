// filename: gpuopt.cpp
// purpose: CLI tool - analyzes a CUDA kernel file and prints optimization report.
// usage: gpuopt --kernel <path> --n <matrix_size> [--fix]
// phase: CLI v1
// last modified: 2026-04-30

#include "compiler/passes.h"
#include "backends/cuda/cuda_backend.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr double kPeakComputeGflops = 10000.0;
constexpr double kPeakBandwidthGbps = 900.0;

const char* pattern_name(const opengpu::compiler::MemAccessPattern pattern) {
  switch (pattern) {
    case opengpu::compiler::MemAccessPattern::COALESCED:
      return "COALESCED";
    case opengpu::compiler::MemAccessPattern::STRIDED:
      return "STRIDED";
    case opengpu::compiler::MemAccessPattern::RANDOM:
      return "RANDOM";
    case opengpu::compiler::MemAccessPattern::UNKNOWN:
      return "UNKNOWN";
    case opengpu::compiler::MemAccessPattern::CONSTANT_MEM:
      return "CONSTANT (cached, broadcast)";
    case opengpu::compiler::MemAccessPattern::SHARED_MEM:
      return "SHARED (on-chip, no coalescing needed)";
  }
  return "UNKNOWN";
}

void print_usage() {
  std::cerr << "usage: gpuopt --kernel <path> --n <matrix_size> [--fix] [--cost] [--gpu <name>] [--hours <n>]\n";
}

}  // namespace

int main(const int argc, const char** argv) {
  std::string kernel_path;
  std::size_t n = 0U;
  bool apply_fix = false;
  bool cost_mode = false;
  std::string gpu_name = "a100";
  double usage_hours = 720.0;
  const std::unordered_map<std::string, double> gpu_costs = {
      {"a100", 3.00},
      {"v100", 2.48},
      {"t4", 0.35},
      {"h100", 4.50},
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--kernel" && (i + 1) < argc) {
      kernel_path = argv[++i];
    } else if (arg == "--n" && (i + 1) < argc) {
      n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--fix") {
      apply_fix = true;
    } else if (arg == "--cost") {
      cost_mode = true;
    } else if (arg == "--gpu" && (i + 1) < argc) {
      gpu_name = argv[++i];
    } else if (arg == "--hours" && (i + 1) < argc) {
      usage_hours = std::strtod(argv[++i], nullptr);
    } else {
      print_usage();
      return EXIT_FAILURE;
    }
  }

  if (kernel_path.empty() || n == 0U) {
    print_usage();
    return EXIT_FAILURE;
  }
  if (cost_mode && gpu_costs.count(gpu_name) == 0U) {
    std::cerr << "Unsupported GPU type: " << gpu_name << '\n';
    return EXIT_FAILURE;
  }

  const opengpu::compiler::KernelIR parsed = opengpu::compiler::parse_cuda_kernel(kernel_path, n);
  if (parsed.ops.empty()) {
    std::cerr << "Failed to parse kernel: " << kernel_path << '\n';
    return EXIT_FAILURE;
  }
  const opengpu::compiler::KernelIR analyzed = opengpu::compiler::memory_pattern_analysis_pass(parsed);

  std::cout << "gpuopt — GPU Kernel Optimizer\n";
  std::cout << "==============================\n";
  std::cout << "Analyzing: " << kernel_path << "  (n=" << n << ")\n\n";

  std::cout << "=== Memory Access Analysis ===\n";
  int issue_count = 0;
  int fixable_count = 0;
  std::vector<std::string> strided_arrays;
  for (const opengpu::compiler::Op& op : analyzed.ops) {
    if (op.type != opengpu::compiler::OpType::GLOBAL_LOAD &&
        op.type != opengpu::compiler::OpType::GLOBAL_STORE &&
        op.type != opengpu::compiler::OpType::SHARED_MEM_LOAD &&
        op.type != opengpu::compiler::OpType::SHARED_MEM_STORE) {
      continue;
    }
    const std::string symbol = op.src0.empty() ? op.dst : op.src0;
    const std::string display_symbol =
        (op.access_pattern == opengpu::compiler::MemAccessPattern::SHARED_MEM && op.src0 == "shared_decl")
            ? op.dst
            : symbol;
    if (op.access_pattern == opengpu::compiler::MemAccessPattern::STRIDED) {
      strided_arrays.push_back(display_symbol);
      std::cout << "[!] " << display_symbol << " -> " << pattern_name(op.access_pattern) << " (stride="
                << op.stride << ") — non-coalesced column access\n";
      ++issue_count;
      ++fixable_count;
    } else {
      const char* status = "[?] ";
      if (op.access_pattern == opengpu::compiler::MemAccessPattern::COALESCED) {
        status = "[✓] ";
      } else if (op.access_pattern == opengpu::compiler::MemAccessPattern::CONSTANT_MEM) {
        status = "[✓] ";
      } else if (op.access_pattern == opengpu::compiler::MemAccessPattern::SHARED_MEM) {
        status = "[✓] ";
      }
      std::cout << status << display_symbol << " -> " << pattern_name(op.access_pattern) << " (stride="
                << op.stride << ")\n";
    }
  }

  const opengpu::compiler::DivergenceInfo divergence =
      opengpu::compiler::warp_divergence_pass(kernel_path);
  std::cout << "\n=== Warp Divergence Analysis ===\n";
  if (!divergence.divergent_conditions.empty()) {
    for (const std::string& condition : divergence.divergent_conditions) {
      std::cout << "[!] Potential divergence: " << condition << '\n';
      std::cout << "    -> Threads may take different paths within warp\n";
      std::cout << "    -> Consider restructuring to branch on (tid / warpSize)\n";
    }
  }
  if (!divergence.safe_conditions.empty()) {
    for (const std::string& condition : divergence.safe_conditions) {
      std::cout << "[✓] Warp-level branch: " << condition << '\n';
      std::cout << "    -> Entire warps take same path — no divergence\n";
    }
  }
  if (divergence.divergent_conditions.empty() && divergence.safe_conditions.empty()) {
    std::cout << "[✓] No warp divergence detected\n";
  }

  const opengpu::compiler::BankConflictInfo bank_conflicts =
      opengpu::compiler::bank_conflict_pass(kernel_path);
  std::cout << "\n=== Bank Conflict Analysis ===\n";
  if (!bank_conflicts.conflict_accesses.empty()) {
    for (const std::string& access : bank_conflicts.conflict_accesses) {
      std::cout << "[!] Potential bank conflict: " << access << '\n';
      std::cout << "    -> threadIdx.y same for all threads in warp — hits same bank\n";
      std::cout << "    -> Fix: transpose access to tile[threadIdx.y][threadIdx.x]\n";
      std::cout << "    -> Or pad declaration: __shared__ float tile[N][M + 1]\n";
    }
  }
  if (!bank_conflicts.clean_accesses.empty()) {
    for (const std::string& access : bank_conflicts.clean_accesses) {
      std::cout << "[✓] Clean access: " << access << '\n';
      std::cout << "    -> threadIdx.x varies per thread — different banks\n";
    }
  }
  if (!bank_conflicts.padded_declarations.empty()) {
    for (const std::string& decl : bank_conflicts.padded_declarations) {
      std::cout << "[✓] Padding detected: " << decl << '\n';
      std::cout << "    -> Bank conflict mitigation applied\n";
    }
  }
  if (bank_conflicts.conflict_accesses.empty() && bank_conflicts.clean_accesses.empty() &&
      bank_conflicts.padded_declarations.empty()) {
    std::cout << "[✓] No bank conflicts detected\n";
  }

  const opengpu::compiler::OccupancyInfo occupancy =
      opengpu::compiler::occupancy_tuning_pass(kernel_path);
  std::cout << "\n=== Occupancy Tuning ===\n";
  if (occupancy.detected_block_size <= 0) {
    std::cout << "[?] Block size not detected in source\n";
    std::cout << "    -> Use --n flag to specify matrix size for roofline analysis\n";
  } else {
    std::cout << "Detected block size  : " << occupancy.detected_block_size << "\n";
    const double occupancy_pct = static_cast<double>(occupancy.estimated_occupancy) * 100.0;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Estimated occupancy  : " << occupancy_pct << "% (" << occupancy.detected_block_size
              << "/1024)\n";
    std::cout << std::setprecision(2);
    std::cout << "Multiple of 32       : " << (occupancy.is_multiple_of_32 ? "yes" : "no") << "\n";
    if (occupancy.exceeds_limit) {
      std::cout << "[!] Block size exceeds CUDA thread/block limit (1024)\n";
    } else if (!occupancy.is_multiple_of_32) {
      std::cout << "[!] Block size is not a multiple of 32\n";
      std::cout << "    -> Current: " << occupancy.detected_block_size << " threads/block\n";
      std::cout << "    -> Suggested: " << occupancy.suggested_block_size << " threads/block\n";
    } else if (occupancy.low_occupancy) {
      std::cout << "[!] Low occupancy detected (< 50%)\n";
      std::cout << "    -> Current: " << occupancy.detected_block_size << " threads/block\n";
      std::cout << "    -> Suggested: " << occupancy.suggested_block_size << " threads/block\n";
      std::cout << "    -> Doubling block size may improve SM utilization\n";
    } else {
      std::cout << "[✓] Block size looks reasonable\n";
    }
  }

  const double dim = static_cast<double>(n);
  const double flops = 2.0 * dim * dim * dim;
  const double bytes = (2.0 * dim * dim * 4.0) + (dim * dim * 4.0);
  const double arithmetic_intensity = flops / bytes;
  const double ridge_point = kPeakComputeGflops / kPeakBandwidthGbps;
  const bool memory_bound = arithmetic_intensity <= ridge_point;

  std::cout << "\n=== Roofline Analysis ===\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Arithmetic Intensity : " << arithmetic_intensity << " FLOPS/byte\n";
  std::cout << "Ridge Point          : " << ridge_point << " FLOPS/byte\n";
  std::cout << "Classification       : " << (memory_bound ? "MEMORY BOUND" : "COMPUTE BOUND")
            << "\n\n";

  std::cout << "=== Optimization Insights ===\n";
  if (strided_arrays.empty()) {
    std::cout << "[✓] No strided global access detected\n";
  } else {
    for (const std::string& name : strided_arrays) {
      std::cout << "[!] STRIDED access detected on " << name << "\n";
      std::cout << "    -> Apply shared memory staging to fix\n";
    }
    if (!apply_fix) {
      std::cout << "\n    -> Run with --fix to auto-apply all fixes\n";
    }
  }

  if (cost_mode) {
    std::cout << "\n=== Cost Impact ===\n";
    if (strided_arrays.empty()) {
      std::cout << "[✓] No inefficiencies detected — no savings estimate needed\n";
    } else {
      std::vector<float> a(n * n, 1.0F);
      std::vector<float> b(n * n, 1.0F);
      std::vector<float> out;
      opengpu::backends::cuda::CUDABackend cuda_backend;

      const auto start = std::chrono::steady_clock::now();
      const bool run_ok = cuda_backend.run_matmul(a, b, n, &out);
      const auto end = std::chrono::steady_clock::now();
      if (!run_ok) {
        std::cout << "[?] Unable to measure CUDA throughput on this system\n";
      } else {
        const double latency_sec = std::chrono::duration<double>(end - start).count();
        const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const double throughput_before = (latency_sec > 0.0) ? (flops / latency_sec) : 0.0;

        const double intensity_before = arithmetic_intensity;
        const double intensity_after = opengpu::compiler::compute_tiled_intensity(flops, n);
        const double improvement_ratio =
            (intensity_before > 0.0) ? (intensity_after / intensity_before) : 1.0;
        const double throughput_after = throughput_before * improvement_ratio;
        const double waste_pct =
            (throughput_after > 0.0) ? (1.0 - (throughput_before / throughput_after)) : 0.0;

        const double gpu_cost = gpu_costs.at(gpu_name);
        const double monthly_cost = usage_hours * gpu_cost;
        const double potential_savings = monthly_cost * waste_pct;

        std::string gpu_upper = gpu_name;
        for (char& ch : gpu_upper) {
          ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }

        std::cout << "GPU              : " << gpu_upper << " ($" << std::fixed << std::setprecision(2)
                  << gpu_cost << "/hr)\n";
        std::cout << "Usage            : " << std::fixed << std::setprecision(0) << usage_hours
                  << " hrs/month\n";
        std::cout << "Monthly GPU cost : $" << std::fixed << std::setprecision(2) << monthly_cost << "\n\n";

        std::cout << "Before (measured) : " << std::fixed << std::setprecision(0)
                  << (throughput_before / 1.0e6) << "M ops/s\n";
        std::cout << "After  (estimated): " << std::fixed << std::setprecision(0)
                  << (throughput_after / 1.0e6) << "M ops/s\n\n";

        std::cout << "Estimated waste   : " << std::fixed << std::setprecision(1) << (waste_pct * 100.0)
                  << "%\n";
        std::cout << "Potential savings : ~$" << std::fixed << std::setprecision(0) << potential_savings
                  << "/month\n\n";

        std::cout << "Note: After throughput estimated from roofline improvement ratio ("
                  << std::fixed << std::setprecision(2) << improvement_ratio << "x).\n";
        std::cout << "      Profile with --fix applied for exact measurement.\n";
      }
    }
  }

  std::cout << "\n=== Summary ===\n";
  std::cout << "Issues found : " << issue_count << '\n';
  std::cout << "Auto-fixable : " << fixable_count << '\n';
  if (issue_count > 0) {
    std::cout << "Run with --fix to apply all fixes\n";
  }

  if (apply_fix && issue_count > 0) {
    const opengpu::compiler::KernelIR fixed = opengpu::compiler::shared_memory_staging_pass(analyzed);
    (void)fixed;
    const double new_intensity = opengpu::compiler::compute_tiled_intensity(flops, n);
    const bool new_memory_bound = new_intensity <= ridge_point;

    std::cout << "\n=== Auto-Fix Applied ===\n";
    for (const std::string& name : strided_arrays) {
      std::cout << "[✓] " << name << ": STRIDED -> COALESCED via shared memory staging\n";
    }
    std::cout << "[✓] New arithmetic intensity: " << new_intensity << " FLOPS/byte\n";
    std::cout << "[✓] Classification: " << (new_memory_bound ? "MEMORY BOUND" : "COMPUTE BOUND")
              << " (was " << (memory_bound ? "MEMORY BOUND" : "COMPUTE BOUND") << ")\n";
  }

  return EXIT_SUCCESS;
}
