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

enum class OpType { LOAD, STORE, MUL, ADD, TILE };

struct Op {
  OpType type;
  std::string dst;
  std::string src0;
  std::string src1;
  std::size_t tile_size;
};

struct KernelIR {
  std::string name;
  std::vector<Op> ops;
};

/**
 * @brief Builds a naive matmul IR sequence.
 * @param None.
 * @return KernelIR containing load/mul/add/store operations.
 * @sideeffects None.
 */
KernelIR build_matmul_ir();

}  // namespace opengpu::compiler

#endif  // OPENGPU_LAB_COMPILER_IR_H_
