// filename: passes.h
// purpose: Declares compiler optimization and analysis passes on KernelIR.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#ifndef OPENGPU_LAB_COMPILER_PASSES_H_
#define OPENGPU_LAB_COMPILER_PASSES_H_

#include <cstddef>
#include <string>

#include "compiler/ir.h"

namespace opengpu::compiler {

/**
 * @brief Inserts TILE ops before each MUL op.
 * @param kernel Input kernel IR.
 * @param tile_size Tile size to annotate on inserted TILE ops.
 * @return Transformed IR with TILE operations inserted.
 * @sideeffects None.
 */
KernelIR loop_tiling_pass(const KernelIR& kernel, std::size_t tile_size);

/**
 * @brief Analyzes whether all LOAD ops are coalesced from TILE context.
 * @param kernel Input kernel IR.
 * @return True if all LOAD ops have preceding TILE op with multiple-of-32 tile.
 * @sideeffects None.
 */
bool memory_coalescing_pass(const KernelIR& kernel);

/**
 * @brief Rewrites TILE op sizes to nearest coalesced multiple of 32.
 * @param kernel Input kernel IR.
 * @return Rewritten IR with TILE sizes auto-fixed for coalescing.
 * @sideeffects None.
 */
KernelIR auto_coalescing_fix_pass(const KernelIR& kernel);

/**
 * @brief Classifies global memory op access patterns from stride annotation.
 * @param kernel Input kernel IR.
 * @return IR annotated with access_pattern for GLOBAL_LOAD/GLOBAL_STORE ops.
 * @sideeffects None.
 */
KernelIR memory_pattern_analysis_pass(const KernelIR& kernel);

/**
 * @brief Parses CUDA kernel source and builds memory-pattern-annotated IR.
 * @param filepath Path to .cu source file.
 * @param n Matrix dimension used for strided access inference.
 * @return Parsed and annotated KernelIR for global load/store operations.
 * @sideeffects Reads kernel source file from disk.
 */
KernelIR parse_cuda_kernel(const std::string& filepath, std::size_t n);

/**
 * @brief Rewrites STRIDED global loads to shared-memory staged coalesced loads.
 * @param kernel Input kernel IR.
 * @return Rewritten IR with staging ops inserted and global loads updated.
 * @sideeffects None.
 */
KernelIR shared_memory_staging_pass(const KernelIR& kernel);

/**
 * @brief Computes arithmetic intensity after shared-memory staging.
 * @param flops Total floating-point operations.
 * @param n Matrix dimension.
 * @return New arithmetic intensity in FLOPS/byte.
 * @sideeffects None.
 */
double compute_tiled_intensity(double flops, std::size_t n);

}  // namespace opengpu::compiler

#endif  // OPENGPU_LAB_COMPILER_PASSES_H_
