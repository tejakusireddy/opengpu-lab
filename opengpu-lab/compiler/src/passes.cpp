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

  bool saw_shared_decl_as = false;
  bool saw_shared_decl_bs = false;
  bool saw_a_load = false;
  bool saw_b_load = false;
  bool saw_c_store = false;
  bool saw_tiled_a_load = false;
  bool saw_tiled_b_load = false;
  bool saw_shared_compute_as = false;
  bool saw_shared_compute_bs = false;
  std::string line;
  while (std::getline(cuda_file, line)) {
    if (!saw_shared_decl_as && line.find("__shared__ float As[") != std::string::npos) {
      saw_shared_decl_as = true;
    }
    if (!saw_shared_decl_bs && line.find("__shared__ float Bs[") != std::string::npos) {
      saw_shared_decl_bs = true;
    }
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
    if (!saw_tiled_a_load && line.find("As[ty][tx] = A[") != std::string::npos &&
        line.find("+ tx") != std::string::npos) {
      saw_tiled_a_load = true;
    }
    if (!saw_tiled_b_load && line.find("Bs[ty][tx] = B[") != std::string::npos &&
        line.find("+ tx") != std::string::npos) {
      saw_tiled_b_load = true;
    }
    if (!saw_shared_compute_as && line.find("As[ty][k]") != std::string::npos) {
      saw_shared_compute_as = true;
    }
    if (!saw_shared_compute_bs && line.find("Bs[k][tx]") != std::string::npos) {
      saw_shared_compute_bs = true;
    }
  }

  KernelIR parsed{};
  parsed.name = "parsed_cuda_kernel";
  if (saw_shared_decl_as) {
    parsed.ops.push_back(
        Op{OpType::SHARED_MEM_LOAD, "As", "shared_decl", "", 0U, MemAccessPattern::COALESCED, 1U});
  }
  if (saw_shared_decl_bs) {
    parsed.ops.push_back(
        Op{OpType::SHARED_MEM_LOAD, "Bs", "shared_decl", "", 0U, MemAccessPattern::COALESCED, 1U});
  }
  if (saw_tiled_a_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "As_tile", "A", "", 0U, MemAccessPattern::UNKNOWN, 1U});
  } else if (saw_a_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "a_reg", "a_device", "", 0U, MemAccessPattern::UNKNOWN, 1U});
  }
  if (saw_tiled_b_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "Bs_tile", "B", "", 0U, MemAccessPattern::UNKNOWN, 1U});
  } else if (saw_b_load) {
    parsed.ops.push_back(
        Op{OpType::GLOBAL_LOAD, "b_reg", "b_device", "", 0U, MemAccessPattern::UNKNOWN, n});
  }
  if (saw_shared_compute_as) {
    parsed.ops.push_back(
        Op{OpType::SHARED_MEM_LOAD, "As_compute", "As[ty][k]", "", 0U, MemAccessPattern::COALESCED, 1U});
  }
  if (saw_shared_compute_bs) {
    parsed.ops.push_back(
        Op{OpType::SHARED_MEM_LOAD, "Bs_compute", "Bs[k][tx]", "", 0U, MemAccessPattern::COALESCED, 1U});
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

KernelIR shared_memory_staging_pass(const KernelIR& kernel) {
  KernelIR rewritten{};
  rewritten.name = kernel.name;
  for (Op op : kernel.ops) {
    if (op.type == OpType::GLOBAL_LOAD && op.access_pattern == MemAccessPattern::STRIDED) {
      rewritten.ops.push_back(
          Op{OpType::SHARED_MEM_LOAD, op.dst + "_staging", op.src0, "", 0U,
             MemAccessPattern::COALESCED, 1U});
      op.stride = 1U;
      op.access_pattern = MemAccessPattern::COALESCED;
    }
    rewritten.ops.push_back(op);
  }
  return rewritten;
}

double compute_tiled_intensity(const double flops, const std::size_t n) {
  const double dim = static_cast<double>(n);
  const double new_bytes = 2.0 * dim * dim * 4.0;
  return flops / new_bytes;
}

}  // namespace opengpu::compiler
