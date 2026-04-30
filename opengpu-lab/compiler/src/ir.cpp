// filename: ir.cpp
// purpose: Implements KernelIR builder for naive matmul representation.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/ir.h"

namespace opengpu::compiler {

KernelIR build_matmul_ir() {
  KernelIR kernel{};
  kernel.name = "matmul";
  kernel.ops = {
      Op{OpType::LOAD, "a_reg", "a", "", 0U},
      Op{OpType::LOAD, "b_reg", "b", "", 0U},
      Op{OpType::MUL, "tmp", "a_reg", "b_reg", 0U},
      Op{OpType::ADD, "c", "c", "tmp", 0U},
      Op{OpType::STORE, "c", "c", "", 0U},
  };
  return kernel;
}

}  // namespace opengpu::compiler
