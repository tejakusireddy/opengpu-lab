// filename: test_compiler.cpp
// purpose: Validates compiler IR build, tiling pass, and coalescing analysis.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/ir.h"
#include "compiler/passes.h"
#include "profiler/profiler.h"

#include <cstdlib>
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

  const std::string kCompilerInsight = "Compiler analysis: memory access pattern is not coalesced";
  if (report_32.find(kCompilerInsight) != std::string::npos) {
    std::cerr << "Compiler coalescing insight unexpectedly fired for tile_size=32\n";
    return EXIT_FAILURE;
  }
  if (report_48.find(kCompilerInsight) == std::string::npos) {
    std::cerr << "Compiler coalescing insight did not fire for tile_size=48\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
