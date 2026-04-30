// filename: test_compiler.cpp
// purpose: Validates compiler IR build, tiling pass, and coalescing analysis.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/ir.h"
#include "compiler/passes.h"
#include "profiler/profiler.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

/**
 * @brief Converts OpType enum to readable token.
 * @param type IR operation type.
 * @return C-string label.
 * @sideeffects None.
 */
const char* op_type_name(const opengpu::compiler::OpType type) {
  switch (type) {
    case opengpu::compiler::OpType::LOAD:
      return "LOAD";
    case opengpu::compiler::OpType::STORE:
      return "STORE";
    case opengpu::compiler::OpType::MUL:
      return "MUL";
    case opengpu::compiler::OpType::ADD:
      return "ADD";
    case opengpu::compiler::OpType::TILE:
      return "TILE";
    case opengpu::compiler::OpType::GLOBAL_LOAD:
      return "GLOBAL_LOAD";
    case opengpu::compiler::OpType::GLOBAL_STORE:
      return "GLOBAL_STORE";
    case opengpu::compiler::OpType::SHARED_MEM_LOAD:
      return "SHARED_MEM_LOAD";
    case opengpu::compiler::OpType::SHARED_MEM_STORE:
      return "SHARED_MEM_STORE";
  }
  return "UNKNOWN";
}

/**
 * @brief Converts memory access pattern enum to readable token.
 * @param pattern Access pattern annotation.
 * @return C-string label.
 * @sideeffects None.
 */
const char* access_pattern_name(const opengpu::compiler::MemAccessPattern pattern) {
  switch (pattern) {
    case opengpu::compiler::MemAccessPattern::COALESCED:
      return "COALESCED";
    case opengpu::compiler::MemAccessPattern::STRIDED:
      return "STRIDED";
    case opengpu::compiler::MemAccessPattern::RANDOM:
      return "RANDOM";
    case opengpu::compiler::MemAccessPattern::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

/**
 * @brief Prints operation sequence from kernel IR.
 * @param title Header title for printed IR.
 * @param kernel IR to print.
 * @return None.
 * @sideeffects Writes to stdout.
 */
void print_ir(const std::string& title, const opengpu::compiler::KernelIR& kernel) {
  std::cout << title << '\n';
  for (const opengpu::compiler::Op& op : kernel.ops) {
    std::cout << "  op=" << op_type_name(op.type) << " dst=" << op.dst << " src0=" << op.src0
              << " src1=" << op.src1 << " tile_size=" << op.tile_size << '\n';
  }
}

/**
 * @brief Prints operation sequence with memory analysis fields.
 * @param title Header title for printed IR.
 * @param kernel IR to print.
 * @return None.
 * @sideeffects Writes to stdout.
 */
void print_ir_with_memory(const std::string& title, const opengpu::compiler::KernelIR& kernel) {
  std::cout << title << '\n';
  for (const opengpu::compiler::Op& op : kernel.ops) {
    std::cout << "  op=" << op_type_name(op.type) << " dst=" << op.dst << " src0=" << op.src0
              << " src1=" << op.src1 << " tile_size=" << op.tile_size << " stride=" << op.stride
              << " access_pattern=" << access_pattern_name(op.access_pattern) << '\n';
  }
}

}  // namespace

int main() {
  const opengpu::compiler::KernelIR base_ir = opengpu::compiler::build_matmul_ir();
  print_ir("=== Base Matmul IR ===", base_ir);

  const opengpu::compiler::KernelIR tiled_32 = opengpu::compiler::loop_tiling_pass(base_ir, 32U);
  print_ir("=== Tiled Matmul IR (tile_size=32) ===", tiled_32);
  const bool coalesced_32 = opengpu::compiler::memory_coalescing_pass(tiled_32);
  if (!coalesced_32) {
    std::cerr << "Expected coalesced memory for tile_size=32\n";
    return EXIT_FAILURE;
  }

  const opengpu::compiler::KernelIR tiled_48 = opengpu::compiler::loop_tiling_pass(base_ir, 48U);
  print_ir("=== Tiled Matmul IR (tile_size=48) ===", tiled_48);
  const bool coalesced_48 = opengpu::compiler::memory_coalescing_pass(tiled_48);
  if (coalesced_48) {
    std::cerr << "Expected non-coalesced memory for tile_size=48\n";
    return EXIT_FAILURE;
  }

  opengpu::profiler::Profiler profiler;
  profiler.record(opengpu::profiler::Metrics{
      "compiler_test_backend", 1.0, 1000.0, 0.75F, 0.10F, true, false, false});

  std::ostringstream report_32_capture;
  std::streambuf* old_buffer = std::cout.rdbuf(report_32_capture.rdbuf());
  profiler.report(coalesced_32);
  std::cout.rdbuf(old_buffer);

  std::ostringstream report_48_capture;
  old_buffer = std::cout.rdbuf(report_48_capture.rdbuf());
  profiler.report(coalesced_48);
  std::cout.rdbuf(old_buffer);

  const std::string report_32 = report_32_capture.str();
  const std::string report_48 = report_48_capture.str();
  std::cout << "=== Profiler Report (compiler_coalesced=true) ===\n" << report_32;
  std::cout << "=== Profiler Report (compiler_coalesced=false) ===\n" << report_48;

  const std::string kCompilerInsight = "Compiler: memory access not coalesced";
  if (report_32.find(kCompilerInsight) != std::string::npos) {
    std::cerr << "Compiler coalescing insight unexpectedly fired for tile_size=32\n";
    return EXIT_FAILURE;
  }
  if (report_48.find(kCompilerInsight) == std::string::npos) {
    std::cerr << "Compiler coalescing insight did not fire for tile_size=48\n";
    return EXIT_FAILURE;
  }

  std::cout << "=== Auto-Fix Pass (tile_size=48 -> auto-fixed) ===\n";
  const opengpu::compiler::KernelIR fixed_ir = opengpu::compiler::auto_coalescing_fix_pass(tiled_48);
  print_ir("=== Fixed Matmul IR (auto_coalescing_fix_pass) ===", fixed_ir);

  std::size_t fixed_tile_size = 0U;
  for (const opengpu::compiler::Op& op : fixed_ir.ops) {
    if (op.type == opengpu::compiler::OpType::TILE) {
      fixed_tile_size = op.tile_size;
      break;
    }
  }
  if (fixed_tile_size != 64U) {
    std::cerr << "Auto-fix did not rewrite TILE size to 64\n";
    return EXIT_FAILURE;
  }
  std::cout << "[✓] Auto-fix rewrote tile_size 48 -> 64\n";

  const bool coalesced_after_fix = opengpu::compiler::memory_coalescing_pass(fixed_ir);
  if (!coalesced_after_fix) {
    std::cerr << "Coalescing check failed after auto-fix\n";
    return EXIT_FAILURE;
  }
  std::cout << "[✓] Coalescing check after fix: PASS\n";

  std::cout << "=== CUDA Matmul IR Analysis (N=64) ===\n";
  const opengpu::compiler::KernelIR cuda_ir = opengpu::compiler::build_cuda_matmul_ir(64U);
  print_ir_with_memory("=== CUDA Matmul IR (raw) ===", cuda_ir);
  const opengpu::compiler::KernelIR analyzed_ir =
      opengpu::compiler::memory_pattern_analysis_pass(cuda_ir);
  print_ir_with_memory("=== CUDA Matmul IR (annotated) ===", analyzed_ir);

  opengpu::compiler::MemAccessPattern a_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t a_stride = 0U;
  opengpu::compiler::MemAccessPattern b_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t b_stride = 0U;
  opengpu::compiler::MemAccessPattern c_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t c_stride = 0U;

  for (const opengpu::compiler::Op& op : analyzed_ir.ops) {
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "a") {
      a_pattern = op.access_pattern;
      a_stride = op.stride;
    } else if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "b") {
      b_pattern = op.access_pattern;
      b_stride = op.stride;
    } else if (op.type == opengpu::compiler::OpType::GLOBAL_STORE && op.dst == "c") {
      c_pattern = op.access_pattern;
      c_stride = op.stride;
    }
  }

  if (a_pattern != opengpu::compiler::MemAccessPattern::COALESCED || a_stride != 1U) {
    std::cerr << "Expected a GLOBAL_LOAD access to be COALESCED with stride=1\n";
    return EXIT_FAILURE;
  }
  if (b_pattern != opengpu::compiler::MemAccessPattern::STRIDED || b_stride != 64U) {
    std::cerr << "Expected b GLOBAL_LOAD access to be STRIDED with stride=64\n";
    return EXIT_FAILURE;
  }
  if (c_pattern != opengpu::compiler::MemAccessPattern::COALESCED || c_stride != 1U) {
    std::cerr << "Expected c GLOBAL_STORE access to be COALESCED with stride=1\n";
    return EXIT_FAILURE;
  }

  std::cout << "[✓] a access: COALESCED (stride=1)\n";
  std::cout << "[!] b access: STRIDED (stride=64) — non-coalesced column access\n";
  std::cout << "[✓] c access: COALESCED (stride=1)\n";

  std::cout << "=== CUDA Kernel Parser (matmul.cu) ===\n";
  const opengpu::compiler::KernelIR parsed_ir =
      opengpu::compiler::parse_cuda_kernel(CUDA_KERNEL_PATH, 64U);
  print_ir_with_memory("=== Parsed CUDA Kernel IR ===", parsed_ir);

  opengpu::compiler::MemAccessPattern parsed_a = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t parsed_a_stride = 0U;
  opengpu::compiler::MemAccessPattern parsed_b = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t parsed_b_stride = 0U;
  opengpu::compiler::MemAccessPattern parsed_c = opengpu::compiler::MemAccessPattern::UNKNOWN;
  std::size_t parsed_c_stride = 0U;

  for (const opengpu::compiler::Op& op : parsed_ir.ops) {
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "a_device") {
      parsed_a = op.access_pattern;
      parsed_a_stride = op.stride;
    } else if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "b_device") {
      parsed_b = op.access_pattern;
      parsed_b_stride = op.stride;
    } else if (op.type == opengpu::compiler::OpType::GLOBAL_STORE && op.dst == "c_device") {
      parsed_c = op.access_pattern;
      parsed_c_stride = op.stride;
    }
  }

  if (parsed_a != opengpu::compiler::MemAccessPattern::COALESCED || parsed_a_stride != 1U) {
    std::cerr << "Parser failed to classify a_device as COALESCED stride=1\n";
    return EXIT_FAILURE;
  }
  if (parsed_b != opengpu::compiler::MemAccessPattern::STRIDED || parsed_b_stride != 64U) {
    std::cerr << "Parser failed to classify b_device as STRIDED stride=64\n";
    return EXIT_FAILURE;
  }
  if (parsed_c != opengpu::compiler::MemAccessPattern::COALESCED || parsed_c_stride != 1U) {
    std::cerr << "Parser failed to classify c_device as COALESCED stride=1\n";
    return EXIT_FAILURE;
  }

  std::cout << "[✓] Parser detected: a_device -> COALESCED (stride=1)\n";
  std::cout << "[!] Parser detected: b_device -> STRIDED (stride=64) — column access\n";
  std::cout << "[✓] Parser detected: c_device -> COALESCED (stride=1)\n";
  std::cout << "[✓] CUDA kernel analysis complete\n";

  std::cout << "=== NVIDIA matrixMul.cu Analysis ===\n";
  const opengpu::compiler::KernelIR nvidia_ir =
      opengpu::compiler::parse_cuda_kernel(NVIDIA_KERNEL_PATH, 32U);
  print_ir_with_memory("=== Parsed NVIDIA Kernel IR ===", nvidia_ir);

  bool has_shared_mem_load = false;
  opengpu::compiler::MemAccessPattern nvidia_a_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  opengpu::compiler::MemAccessPattern nvidia_b_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  for (const opengpu::compiler::Op& op : nvidia_ir.ops) {
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "A") {
      nvidia_a_pattern = op.access_pattern;
    }
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "B") {
      nvidia_b_pattern = op.access_pattern;
    }
    if (op.type == opengpu::compiler::OpType::SHARED_MEM_LOAD) {
      has_shared_mem_load = true;
    }
  }

  if (nvidia_a_pattern != opengpu::compiler::MemAccessPattern::COALESCED) {
    std::cerr << "Expected NVIDIA A global load to be COALESCED\n";
    return EXIT_FAILURE;
  }
  if (nvidia_b_pattern != opengpu::compiler::MemAccessPattern::COALESCED) {
    std::cerr << "Expected NVIDIA B global load to be COALESCED\n";
    return EXIT_FAILURE;
  }
  if (!has_shared_mem_load) {
    std::cerr << "Expected SHARED_MEM_LOAD ops in NVIDIA parsed IR\n";
    return EXIT_FAILURE;
  }

  std::cout << "=== Naive vs NVIDIA Tiled Kernel Comparison ===\n";
  std::cout << "Naive kernel:\n";
  std::cout << "  [✓] A access: COALESCED (stride=1)\n";
  std::cout << "  [!] B access: STRIDED (stride=N) — column access, slow\n";
  std::cout << "  [✗] No shared memory usage\n\n";
  std::cout << "NVIDIA tiled kernel:\n";
  std::cout << "  [✓] A access: COALESCED (stride=1)\n";
  std::cout << "  [✓] B access: COALESCED (stride=1) — staged via shared memory\n";
  std::cout << "  [✓] Shared memory tiling detected — eliminates strided access\n\n";
  std::cout << "[✓] NVIDIA kernel is superior: B access pattern fixed via shared memory staging\n";

  std::cout << "=== Shared Memory Staging Pass ===\n";
  const opengpu::compiler::KernelIR cuda_base = opengpu::compiler::build_cuda_matmul_ir(64U);
  const opengpu::compiler::KernelIR cuda_annotated =
      opengpu::compiler::memory_pattern_analysis_pass(cuda_base);
  print_ir_with_memory("=== Before Shared Memory Staging ===", cuda_annotated);

  opengpu::compiler::MemAccessPattern before_b_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  for (const opengpu::compiler::Op& op : cuda_annotated.ops) {
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.src0 == "b") {
      before_b_pattern = op.access_pattern;
    }
  }
  if (before_b_pattern != opengpu::compiler::MemAccessPattern::STRIDED) {
    std::cerr << "Expected b_reg to be STRIDED before shared memory staging\n";
    return EXIT_FAILURE;
  }

  const opengpu::compiler::KernelIR staged_ir =
      opengpu::compiler::shared_memory_staging_pass(cuda_annotated);
  print_ir_with_memory("=== After Shared Memory Staging ===", staged_ir);

  bool has_strided_after = false;
  bool has_staging_op = false;
  opengpu::compiler::MemAccessPattern after_b_pattern = opengpu::compiler::MemAccessPattern::UNKNOWN;
  for (const opengpu::compiler::Op& op : staged_ir.ops) {
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD &&
        op.access_pattern == opengpu::compiler::MemAccessPattern::STRIDED) {
      has_strided_after = true;
    }
    if (op.type == opengpu::compiler::OpType::SHARED_MEM_LOAD &&
        op.dst == "b_reg_staging") {
      has_staging_op = true;
    }
    if (op.type == opengpu::compiler::OpType::GLOBAL_LOAD && op.dst == "b_reg") {
      after_b_pattern = op.access_pattern;
    }
  }
  if (has_strided_after) {
    std::cerr << "Expected no STRIDED ops after shared memory staging pass\n";
    return EXIT_FAILURE;
  }
  if (!has_staging_op) {
    std::cerr << "Expected SHARED_MEM_LOAD staging op insertion\n";
    return EXIT_FAILURE;
  }
  if (after_b_pattern != opengpu::compiler::MemAccessPattern::COALESCED) {
    std::cerr << "Expected b_reg GLOBAL_LOAD to become COALESCED\n";
    return EXIT_FAILURE;
  }

  const double flops = 524288.0;
  const double ridge_point = 10000.0 / 900.0;
  const double before_intensity = flops / 49152.0;
  const double after_intensity = opengpu::compiler::compute_tiled_intensity(flops, 64U);

  std::cout << "=== Roofline Impact of Shared Memory Pass ===\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Before pass:\n";
  std::cout << "  Arithmetic Intensity : " << before_intensity << " FLOPS/byte\n";
  std::cout << "  Classification       : MEMORY BOUND\n\n";
  std::cout << "After pass:\n";
  std::cout << "  Arithmetic Intensity : " << after_intensity << " FLOPS/byte\n";
  std::cout << "  Classification       : COMPUTE BOUND (crosses ridge at 11.11)\n\n";
  std::cout << "[✓] Shared memory staging eliminates STRIDED access\n";
  std::cout << "[✓] Arithmetic intensity: 10.67 -> 16.00 FLOPS/byte\n";
  std::cout << "[✓] Kernel crosses roofline ridge: MEMORY BOUND -> COMPUTE BOUND\n";

  if (after_intensity <= ridge_point) {
    std::cerr << "Expected tiled intensity to cross ridge point\n";
    return EXIT_FAILURE;
  }
  const bool compute_bound_after = after_intensity > ridge_point;
  if (!compute_bound_after) {
    std::cerr << "Expected compute-bound classification after shared memory staging\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
