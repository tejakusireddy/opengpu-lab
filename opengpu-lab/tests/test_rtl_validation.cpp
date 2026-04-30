// filename: test_rtl_validation.cpp
// purpose: Verifies RTL signal-level reset and done-cycle behavior in Verilator harness.
// phase: RTL v2
// last modified: 2026-04-30

#include "verilator_harness.h"

#include <cstdlib>
#include <iostream>

int main() {
  constexpr std::size_t kDim = 4U;
  constexpr std::size_t kDoneBound = (kDim * kDim * kDim) + 2U;

  opengpu::verilator::VerilatorHarness harness;

  const bool reset_ok = harness.verify_reset_clears_output(kDim);
  std::cout << "[\u2713] Reset clears output: " << (reset_ok ? "PASS" : "FAIL") << '\n';
  if (!reset_ok) {
    return EXIT_FAILURE;
  }

  const bool done_ok = harness.verify_done_on_correct_cycle(kDim);
  const std::size_t done_cycle = harness.last_done_cycle();
  std::cout << "[\u2713] Done asserted on cycle " << done_cycle << " (bound=" << kDoneBound
            << "): " << (done_ok ? "PASS" : "FAIL") << '\n';
  if (!done_ok) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
