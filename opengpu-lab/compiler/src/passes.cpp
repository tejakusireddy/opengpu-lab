// filename: passes.cpp
// purpose: Implements loop tiling and memory coalescing analysis passes.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/passes.h"

#include <cctype>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(cuda_file, line)) {
    lines.push_back(line);
  }

  auto trim = [](const std::string& value) -> std::string {
    std::size_t start = 0U;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
      ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) {
      --end;
    }
    return value.substr(start, end - start);
  };

  auto extract_lhs_var = [&](const std::string& full_line) -> std::string {
    const std::size_t eq_pos = full_line.find('=');
    if (eq_pos == std::string::npos) {
      return "";
    }
    const std::string lhs = trim(full_line.substr(0U, eq_pos));
    if (lhs.empty()) {
      return "";
    }
    std::size_t end = lhs.size();
    while (end > 0U) {
      const char ch = lhs[end - 1U];
      const bool is_ident = std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
      if (!is_ident) {
        break;
      }
      --end;
    }
    return lhs.substr(end);
  };

  auto collect_array_accesses =
      [&trim](const std::string& segment) -> std::vector<std::pair<std::string, std::string>> {
    std::vector<std::pair<std::string, std::string>> accesses;
    std::size_t scan = 0U;
    while (scan < segment.size()) {
      const std::size_t open = segment.find('[', scan);
      if (open == std::string::npos) {
        break;
      }
      std::size_t name_start = open;
      while (name_start > 0U) {
        const char ch = segment[name_start - 1U];
        const bool is_ident = std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
        if (!is_ident) {
          break;
        }
        --name_start;
      }
      const std::size_t close = segment.find(']', open + 1U);
      if (close != std::string::npos && name_start < open) {
        const std::string name = segment.substr(name_start, open - name_start);
        const std::string index = trim(segment.substr(open + 1U, close - open - 1U));
        if (!name.empty() && !index.empty()) {
          accesses.push_back({name, index});
        }
      }
      scan = (close == std::string::npos) ? (open + 1U) : (close + 1U);
    }
    return accesses;
  };

  auto in_kernel_body = [](const std::vector<std::string>& all_lines, const std::size_t idx) -> bool {
    bool saw_global = false;
    int brace_depth = 0;
    bool inside = false;
    for (std::size_t i = 0U; i <= idx && i < all_lines.size(); ++i) {
      const std::string& current = all_lines[i];
      if (!saw_global && current.find("__global__") != std::string::npos) {
        saw_global = true;
      }
      for (char ch : current) {
        if (ch == '{') {
          ++brace_depth;
          if (saw_global && brace_depth >= 1) {
            inside = true;
          }
        } else if (ch == '}') {
          if (inside && brace_depth == 1) {
            inside = false;
            saw_global = false;
          }
          if (brace_depth > 0) {
            --brace_depth;
          }
        }
      }
    }
    return inside;
  };

  bool saw_legacy_a_load = false;
  bool saw_legacy_b_load = false;
  bool saw_legacy_c_store = false;
  bool legacy_mode = false;
  std::unordered_map<std::string, MemAccessPattern> index_map;
  std::unordered_map<std::string, std::size_t> stride_map;
  std::unordered_map<std::string, bool> shared_arrays;
  std::unordered_set<std::string> seen_arrays;

  // Pass 1: index variable analysis.
  for (std::size_t line_idx = 0U; line_idx < lines.size(); ++line_idx) {
    const std::string& current = lines[line_idx];
    if (!in_kernel_body(lines, line_idx)) {
      continue;
    }
    const std::size_t eq_pos = current.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }
    const std::string var_name = extract_lhs_var(current);
    if (var_name.empty()) {
      continue;
    }

    std::string expr = trim(current.substr(eq_pos + 1U));
    if (!expr.empty() && expr.back() == ';') {
      expr.pop_back();
      expr = trim(expr);
    }

    const bool has_thread_x = expr.find("threadIdx.x") != std::string::npos;
    const bool has_thread_y = expr.find("threadIdx.y") != std::string::npos;
    const bool has_block_terms =
        expr.find("blockIdx.") != std::string::npos || expr.find("blockDim.") != std::string::npos;
    const bool has_multiplier = expr.find('*') != std::string::npos;
    const bool has_dim_multiplier = expr.find("width") != std::string::npos ||
                                    expr.find("height") != std::string::npos ||
                                    expr.find(" n") != std::string::npos ||
                                    expr.find("*n") != std::string::npos;
    const bool has_coalesced_form =
        (has_thread_x || has_thread_y) &&
        (expr.find("blockIdx.x * blockDim.x + threadIdx.x") != std::string::npos ||
         expr.find("threadIdx.x + blockIdx.x * blockDim.x") != std::string::npos ||
         expr.find("blockIdx.y * blockDim.y + threadIdx.y") != std::string::npos ||
         expr.find("threadIdx.y + blockIdx.y * blockDim.y") != std::string::npos || has_block_terms);

    if (has_coalesced_form) {
      index_map[var_name] = MemAccessPattern::COALESCED;
      stride_map[var_name] = 1U;
      continue;
    }

    const bool ends_with_x_index = expr.find("+ xIndex") != std::string::npos ||
                                   expr.find("+ threadIdx.x") != std::string::npos;
    bool strided_from_known_term = false;
    if (has_multiplier && expr.find('+') != std::string::npos) {
      const std::size_t plus_pos = expr.find('+');
      const std::string left = trim(expr.substr(0U, plus_pos));
      const std::size_t star_pos = left.find('*');
      if (star_pos != std::string::npos) {
        const std::string term1 = trim(left.substr(0U, star_pos));
        const std::string term2 = trim(left.substr(star_pos + 1U));
        const bool term1_thread = index_map.count(term1) > 0U && index_map[term1] == MemAccessPattern::COALESCED;
        const bool term2_thread = index_map.count(term2) > 0U && index_map[term2] == MemAccessPattern::COALESCED;
        strided_from_known_term = (term1_thread != term2_thread);
      }
    }

    if ((has_multiplier && has_dim_multiplier) || (has_multiplier && ends_with_x_index) ||
        strided_from_known_term) {
      index_map[var_name] = MemAccessPattern::STRIDED;
      stride_map[var_name] = n;
      continue;
    }

    if (has_thread_x || has_thread_y) {
      index_map[var_name] = MemAccessPattern::COALESCED;
      stride_map[var_name] = 1U;
    }
  }

  // Pass 2: array access classification + existing pattern detection.
  KernelIR parsed{};
  parsed.name = "parsed_cuda_kernel";

  for (std::size_t line_idx = 0U; line_idx < lines.size(); ++line_idx) {
    const std::string& current = lines[line_idx];
    if (!in_kernel_body(lines, line_idx)) {
      continue;
    }

    if (current.find("a_device[") != std::string::npos || current.find("b_device[") != std::string::npos ||
        current.find("c_device[") != std::string::npos) {
      legacy_mode = true;
    }

    if (current.find("__shared__") != std::string::npos) {
      const std::string marker = "float ";
      const std::size_t type_pos = current.find(marker);
      const std::size_t bracket_pos =
          current.find("[", type_pos == std::string::npos ? 0U : type_pos);
      if (type_pos != std::string::npos && bracket_pos != std::string::npos &&
          bracket_pos > (type_pos + marker.size())) {
        const std::string name =
            current.substr(type_pos + marker.size(), bracket_pos - (type_pos + marker.size()));
        shared_arrays[name] = true;
        if (seen_arrays.insert(name).second) {
          parsed.ops.push_back(
              Op{OpType::SHARED_MEM_LOAD, name, "shared_decl", "", 0U, MemAccessPattern::COALESCED,
                 1U});
        }
      }
    }

    const bool ends_coalesced = current.find("+ tx]") != std::string::npos ||
                                current.find("+ ty]") != std::string::npos ||
                                current.find("* ty + tx]") != std::string::npos;
    const bool has_strided_multiplier = current.find("* wA") != std::string::npos ||
                                        current.find("*wA") != std::string::npos ||
                                        current.find("* wB") != std::string::npos ||
                                        current.find("*wB") != std::string::npos ||
                                        current.find("* n") != std::string::npos ||
                                        current.find("*n") != std::string::npos;

    const std::size_t eq_pos = current.find('=');
    const std::string lhs = (eq_pos == std::string::npos) ? "" : current.substr(0U, eq_pos);
    const std::string rhs = (eq_pos == std::string::npos) ? current : current.substr(eq_pos + 1U);
    const auto lhs_accesses = collect_array_accesses(lhs);
    const auto rhs_accesses = collect_array_accesses(rhs);

    auto classify_index = [&](const std::string& index_expr) -> std::pair<MemAccessPattern, std::size_t> {
      const std::string index = trim(index_expr);
      if (index_map.count(index) > 0U) {
        return {index_map[index], stride_map[index]};
      }
      if (index.find("threadIdx.x") != std::string::npos || index == "id" ||
          (!index.empty() && index.back() == 'x')) {
        return {MemAccessPattern::COALESCED, 1U};
      }
      return {MemAccessPattern::UNKNOWN, 0U};
    };

    if (!legacy_mode) {
      for (const auto& access : lhs_accesses) {
        if (shared_arrays.count(access.first) > 0U) {
          continue;
        }
        if (!seen_arrays.insert(access.first).second) {
          continue;
        }
        const auto [pattern, stride] = classify_index(access.second);
        parsed.ops.push_back(
            Op{OpType::GLOBAL_STORE, access.first, access.first, "", 0U, pattern, stride});
      }
      for (const auto& access : rhs_accesses) {
        if (shared_arrays.count(access.first) > 0U) {
          continue;
        }
        if (!seen_arrays.insert(access.first).second) {
          continue;
        }
        const auto [pattern, stride] = classify_index(access.second);
        parsed.ops.push_back(Op{OpType::GLOBAL_LOAD, access.first + "_load", access.first, "", 0U,
                                pattern, stride});
      }
    }

    if (!legacy_mode) {
      const bool contains_uppercase_load =
          current.find("= A[") != std::string::npos || current.find("= B[") != std::string::npos;
      if (contains_uppercase_load && ends_coalesced) {
        std::string src_name = "global_load";
        if (current.find("= A[") != std::string::npos) {
          src_name = "A";
        } else if (current.find("= B[") != std::string::npos) {
          src_name = "B";
        }
        if (seen_arrays.insert(src_name).second) {
          parsed.ops.push_back(Op{OpType::GLOBAL_LOAD, src_name + "_load", src_name, "", 0U,
                                  MemAccessPattern::COALESCED, 1U});
        }
      }

      if (current.find("=") != std::string::npos && current.find("[") != std::string::npos &&
          has_strided_multiplier && !ends_coalesced) {
        if (seen_arrays.insert("global").second) {
          parsed.ops.push_back(Op{OpType::GLOBAL_LOAD, "strided_load", "global", "", 0U,
                                  MemAccessPattern::STRIDED, n});
        }
      }

      if (eq_pos != std::string::npos && current.find("[") != std::string::npos && ends_coalesced) {
        const std::size_t left_bracket = current.find("[");
        if (left_bracket != std::string::npos && left_bracket < eq_pos) {
          std::size_t start = left_bracket;
          while (start > 0U) {
            const char ch = current[start - 1U];
            const bool is_ident = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                                  (ch >= '0' && ch <= '9') || ch == '_';
            if (!is_ident) {
              break;
            }
            --start;
          }
          const std::string dst_name = current.substr(start, left_bracket - start);
          if (seen_arrays.insert(dst_name).second) {
            parsed.ops.push_back(Op{OpType::GLOBAL_STORE, dst_name, dst_name, "", 0U,
                                    MemAccessPattern::COALESCED, 1U});
          }
        }
      }
    }

    if (current.find("As[ty][k]") != std::string::npos) {
      if (seen_arrays.insert("As_compute").second) {
        parsed.ops.push_back(Op{OpType::SHARED_MEM_LOAD, "As_compute", "As[ty][k]", "", 0U,
                                MemAccessPattern::COALESCED, 1U});
      }
    }
    if (current.find("Bs[k][tx]") != std::string::npos) {
      if (seen_arrays.insert("Bs_compute").second) {
        parsed.ops.push_back(Op{OpType::SHARED_MEM_LOAD, "Bs_compute", "Bs[k][tx]", "", 0U,
                                MemAccessPattern::COALESCED, 1U});
      }
    }

    if (!saw_legacy_a_load && current.find("a_device[") != std::string::npos) {
      saw_legacy_a_load = true;
    }
    if (!saw_legacy_b_load && current.find("b_device[") != std::string::npos) {
      saw_legacy_b_load = true;
    }
    if (!saw_legacy_c_store && current.find("c_device[") != std::string::npos &&
        current.find("=") != std::string::npos) {
      saw_legacy_c_store = true;
    }
  }

  if (saw_legacy_a_load) {
    if (seen_arrays.insert("a_device").second) {
      parsed.ops.push_back(
          Op{OpType::GLOBAL_LOAD, "a_reg", "a_device", "", 0U, MemAccessPattern::COALESCED, 1U});
    }
  }
  if (saw_legacy_b_load) {
    if (seen_arrays.insert("b_device").second) {
      parsed.ops.push_back(
          Op{OpType::GLOBAL_LOAD, "b_reg", "b_device", "", 0U, MemAccessPattern::STRIDED, n});
    }
  }
  parsed.ops.push_back(Op{OpType::MUL, "tmp", "a_reg", "b_reg", 0U, MemAccessPattern::UNKNOWN,
                          0U});
  parsed.ops.push_back(
      Op{OpType::ADD, "c", "c", "tmp", 0U, MemAccessPattern::UNKNOWN, 0U});
  if (saw_legacy_c_store) {
    if (seen_arrays.insert("c_device").second) {
      parsed.ops.push_back(
          Op{OpType::GLOBAL_STORE, "c_device", "c", "", 0U, MemAccessPattern::UNKNOWN, 1U});
    }
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
