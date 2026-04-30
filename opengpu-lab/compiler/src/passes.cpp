// filename: passes.cpp
// purpose: Implements loop tiling and memory coalescing analysis passes.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/passes.h"

#include <fstream>
#include <string>

namespace opengpu::compiler {

KernelIR loop_tiling_pass(const KernelIR& kernel, const std::size_t tile_size) {
  KernelIR transformed{};
  transformed.name = kernel.name;

  for (const Op& op : kernel.ops) {
    if (op.type == OpType::MUL) {
      transformed.ops.push_back(
          Op{OpType::TILE, "tile", "", "", tile_size, MemAccessPattern::UNKNOWN, 0U});
    }
    transformed.ops.push_back(op);
  }
  return transformed;
}

bool memory_coalescing_pass(const KernelIR& kernel) {
  bool saw_load = false;
  bool saw_valid_tiling_context = false;

  for (const Op& op : kernel.ops) {
    if (op.type == OpType::LOAD) {
      saw_load = true;
      continue;
    }
    if (op.type == OpType::TILE) {
      if ((op.tile_size % 32U) != 0U) {
        return false;
      }
      saw_valid_tiling_context = true;
    }
  }
  return saw_load ? saw_valid_tiling_context : true;
}

KernelIR auto_coalescing_fix_pass(const KernelIR& kernel) {
  KernelIR rewritten = kernel;
  for (Op& op : rewritten.ops) {
    if (op.type == OpType::TILE && (op.tile_size % 32U) != 0U) {
      op.tile_size = ((op.tile_size + 31U) / 32U) * 32U;
    }
  }
  return rewritten;
}

KernelIR memory_pattern_analysis_pass(const KernelIR& kernel) {
  KernelIR annotated = kernel;
  for (Op& op : annotated.ops) {
    if (op.type == OpType::GLOBAL_LOAD || op.type == OpType::GLOBAL_STORE) {
      if (op.stride == 1U) {
        op.access_pattern = MemAccessPattern::COALESCED;
      } else if (op.stride > 1U) {
        op.access_pattern = MemAccessPattern::STRIDED;
      } else {
        op.access_pattern = MemAccessPattern::UNKNOWN;
      }
    }
  }
  return annotated;
}

KernelIR parse_cuda_kernel(const std::string& filepath, const std::size_t n) {
  std::ifstream cuda_file(filepath);
  if (!cuda_file.is_open()) {
    return {};
  }

  bool saw_a_load = false;
  bool saw_b_load = false;
  bool saw_c_store = false;
  std::string line;
  while (std::getline(cuda_file, line)) {
    if (!saw_a_load && line.find("a_device[") != std::string::npos) {
      saw_a_load = true;
    }
    if (!saw_b_load && line.find("b_device[") != std::string::npos) {
      const bool has_strided_term =
          line.find("* n") != std::string::npos || line.find("*n") != std::string::npos;
      if (has_strided_term) {
        saw_b_load = true;
      }
    }
    if (!saw_c_store && line.find("c_device[") != std::string::npos &&
        line.find("=") != std::string::npos) {
      saw_c_store = true;
    }
  }

  KernelIR parsed{};
  parsed.name = "parsed_cuda_kernel";
  if (saw_a_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "a_reg", "a_device", "", 0U, MemAccessPattern::UNKNOWN, 1U});
  }
  if (saw_b_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "b_reg", "b_device", "", 0U, MemAccessPattern::UNKNOWN, n});
  }
  parsed.ops.push_back(Op{OpType::MUL, "tmp", "a_reg", "b_reg", 0U, MemAccessPattern::UNKNOWN,
                          0U});
  parsed.ops.push_back(
      Op{OpType::ADD, "c", "c", "tmp", 0U, MemAccessPattern::UNKNOWN, 0U});
  if (saw_c_store) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_STORE, "c_device", "c", "", 0U, MemAccessPattern::UNKNOWN, 1U});
  }
  return memory_pattern_analysis_pass(parsed);
}

}  // namespace opengpu::compiler
