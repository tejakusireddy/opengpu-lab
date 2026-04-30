// filename: ir.h
// purpose: Defines intermediate representation for compiler kernel passes.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#ifndef OPENGPU_LAB_COMPILER_IR_H_
#define OPENGPU_LAB_COMPILER_IR_H_

#include <cstddef>
#include <string>
#include <vector>

namespace opengpu::compiler {

enum class MemAccessPattern {
  COALESCED,
  STRIDED,
  RANDOM,
  UNKNOWN,
  CONSTANT_MEM,
  SHARED_MEM
};

enum class OpType {
  LOAD,
  STORE,
  MUL,
  ADD,
  TILE,
  GLOBAL_LOAD,
  GLOBAL_STORE,
  SHARED_MEM_LOAD,
  SHARED_MEM_STORE
};

struct Op {
  OpType type;
  std::string dst;
  std::string src0;
  std::string src1;
  std::size_t tile_size;
  MemAccessPattern access_pattern = MemAccessPattern::UNKNOWN;
  std::size_t stride = 0U;
};

struct KernelIR {
  std::string name;
  std::vector<Op> ops;
};

struct DivergenceInfo {
  bool has_divergence;
  std::vector<std::string> divergent_conditions;
  std::vector<std::string> safe_conditions;
};

/**
 * @brief Builds a naive matmul IR sequence.
 * @param None.
 * @return KernelIR containing load/mul/add/store operations.
 * @sideeffects None.
 */
KernelIR build_matmul_ir();

/**
 * @brief Builds a naive CUDA matmul IR with memory stride annotations.
 * @param N Matrix dimension used for strided B-side global loads.
 * @return KernelIR containing CUDA-style global memory operations.
 * @sideeffects None.
 */
KernelIR build_cuda_matmul_ir(std::size_t N);

}  // namespace opengpu::compiler

#endif  // OPENGPU_LAB_COMPILER_IR_H_
