// filename: verilator_harness.cpp
// purpose: Implements Verilator harness for RTL matmul execution.
// phase: Phase 6 - Integration
// last modified: 2026-04-29

#include "verilator_harness.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "Vmatmul_accelerator.h"
#include "verilated.h"

double sc_time_stamp() { return 0.0; }

namespace opengpu::verilator {

namespace {

constexpr std::size_t kModelDim = 4U;
constexpr std::size_t kElemCount = kModelDim * kModelDim;
constexpr std::size_t kMaxSimulationCycles = 2048U;

/**
 * @brief Advances the Verilated model by one full clock cycle.
 * @param model Verilated top module instance.
 * @return None.
 * @sideeffects Updates RTL state and combinational outputs.
 */
void tick(Vmatmul_accelerator* model) {
  model->clk = 0U;
  model->eval();
  model->clk = 1U;
  model->eval();
}

/**
 * @brief Builds deterministic non-zero matrix data for harness validation.
 * @param n Matrix dimension.
 * @param seed Value offset.
 * @return Row-major matrix values.
 * @sideeffects None.
 */
std::vector<float> make_matrix(const std::size_t n, const std::size_t seed) {
  std::vector<float> out(n * n, 0.0F);
  constexpr std::size_t kValueMod = 17U;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<float>(((i + seed) % kValueMod) + 1U);
  }
  return out;
}

/**
 * @brief Executes one 4x4 matrix multiply on the Verilated model.
 * @param a_tile Row-major 4x4 tile A.
 * @param b_tile Row-major 4x4 tile B.
 * @return Row-major 4x4 output tile C.
 * @sideeffects Drives and advances the Verilated model.
 */
std::vector<float> run_tile_4x4(const std::vector<float>& a_tile,
                                const std::vector<float>& b_tile) {
  std::unique_ptr<Vmatmul_accelerator> model = std::make_unique<Vmatmul_accelerator>();
  model->start = 0U;
  model->rst = 1U;
  tick(model.get());
  tick(model.get());
  model->rst = 0U;

  for (std::size_t idx = 0; idx < kElemCount; ++idx) {
    model->a_flat[idx] = static_cast<std::uint32_t>(a_tile[idx]);
    model->b_flat[idx] = static_cast<std::uint32_t>(b_tile[idx]);
  }

  model->start = 1U;
  tick(model.get());
  model->start = 0U;

  for (std::size_t cycle = 0; cycle < kMaxSimulationCycles; ++cycle) {
    tick(model.get());
    if (model->done != 0U) {
      std::vector<float> output(kElemCount, 0.0F);
      for (std::size_t idx = 0; idx < kElemCount; ++idx) {
        output[idx] = static_cast<float>(model->c_flat[idx]);
      }
      return output;
    }
  }
  return {};
}

}  // namespace

std::vector<float> VerilatorHarness::run_matmul(const std::vector<float>& a,
                                                const std::vector<float>& b,
                                                const std::size_t n) const {
  if (n == 0U || (n % kModelDim) != 0U) {
    return {};
  }
  const std::size_t expected = n * n;
  if (a.size() != expected || b.size() != expected) {
    return {};
  }

  std::vector<float> output(expected, 0.0F);
  const std::size_t tile_count = n / kModelDim;

  for (std::size_t tile_row = 0; tile_row < tile_count; ++tile_row) {
    for (std::size_t tile_col = 0; tile_col < tile_count; ++tile_col) {
      std::vector<float> accum_tile(kElemCount, 0.0F);

      for (std::size_t tile_k = 0; tile_k < tile_count; ++tile_k) {
        std::vector<float> a_tile(kElemCount, 0.0F);
        std::vector<float> b_tile(kElemCount, 0.0F);

        for (std::size_t i = 0; i < kModelDim; ++i) {
          for (std::size_t j = 0; j < kModelDim; ++j) {
            const std::size_t a_row = (tile_row * kModelDim) + i;
            const std::size_t a_col = (tile_k * kModelDim) + j;
            const std::size_t b_row = (tile_k * kModelDim) + i;
            const std::size_t b_col = (tile_col * kModelDim) + j;
            a_tile[(i * kModelDim) + j] = a[(a_row * n) + a_col];
            b_tile[(i * kModelDim) + j] = b[(b_row * n) + b_col];
          }
        }

        const std::vector<float> tile_out = run_tile_4x4(a_tile, b_tile);
        if (tile_out.size() != kElemCount) {
          return {};
        }
        for (std::size_t idx = 0; idx < kElemCount; ++idx) {
          accum_tile[idx] += tile_out[idx];
        }
      }

      for (std::size_t i = 0; i < kModelDim; ++i) {
        for (std::size_t j = 0; j < kModelDim; ++j) {
          const std::size_t out_row = (tile_row * kModelDim) + i;
          const std::size_t out_col = (tile_col * kModelDim) + j;
          output[(out_row * n) + out_col] = accum_tile[(i * kModelDim) + j];
        }
      }
    }
  }
  return output;
}

bool VerilatorHarness::verify_reset_clears_output(const std::size_t n) {
  if (n != kModelDim) {
    return false;
  }

  const std::vector<float> a = make_matrix(n, 3U);
  const std::vector<float> b = make_matrix(n, 9U);
  std::unique_ptr<Vmatmul_accelerator> model = std::make_unique<Vmatmul_accelerator>();

  model->start = 0U;
  model->rst = 0U;
  for (std::size_t idx = 0; idx < kElemCount; ++idx) {
    model->a_flat[idx] = static_cast<std::uint32_t>(a[idx]);
    model->b_flat[idx] = static_cast<std::uint32_t>(b[idx]);
    model->c_flat[idx] = 0xFFFFFFFFU;
  }

  model->rst = 1U;
  tick(model.get());
  tick(model.get());
  model->rst = 0U;
  tick(model.get());

  for (std::size_t idx = 0; idx < kElemCount; ++idx) {
    if (model->c_flat[idx] != 0U) {
      return false;
    }
  }
  return true;
}

bool VerilatorHarness::verify_done_on_correct_cycle(const std::size_t n) {
  if (n != kModelDim) {
    last_done_cycle_ = 0U;
    return false;
  }

  const std::vector<float> a = make_matrix(n, 5U);
  const std::vector<float> b = make_matrix(n, 11U);
  std::unique_ptr<Vmatmul_accelerator> model = std::make_unique<Vmatmul_accelerator>();

  model->start = 0U;
  model->rst = 1U;
  tick(model.get());
  tick(model.get());
  model->rst = 0U;

  for (std::size_t idx = 0; idx < kElemCount; ++idx) {
    model->a_flat[idx] = static_cast<std::uint32_t>(a[idx]);
    model->b_flat[idx] = static_cast<std::uint32_t>(b[idx]);
  }

  model->start = 1U;
  tick(model.get());
  model->start = 0U;

  const std::size_t cycle_bound = (n * n * n) + 2U;
  last_done_cycle_ = 0U;
  for (std::size_t cycle = 1U; cycle <= kMaxSimulationCycles; ++cycle) {
    tick(model.get());
    if (model->done != 0U) {
      last_done_cycle_ = cycle;
      break;
    }
  }

  if (last_done_cycle_ == 0U) {
    return false;
  }
  return last_done_cycle_ <= cycle_bound;
}

std::size_t VerilatorHarness::last_done_cycle() const { return last_done_cycle_; }

}  // namespace opengpu::verilator
