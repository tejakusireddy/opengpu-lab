// filename: profiler.h
// purpose: Declares profiler and rule-based optimization insight engine.
// phase: Phase 7 - Profiling
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_PROFILER_PROFILER_H_
#define OPENGPU_LAB_PROFILER_PROFILER_H_

#include <vector>

#include "profiler/metrics.h"

namespace opengpu::profiler {

/**
 * @brief Records and retrieves backend execution metrics.
 * @param None.
 * @return N/A.
 * @sideeffects Internal metric state may be updated.
 */
class Profiler {
 public:
  Profiler() = default;

  /**
   * @brief Stores metrics sample.
   * @param metrics Metric values to persist.
   * @return None.
   * @sideeffects Replaces currently stored metrics.
   */
  void record(const Metrics& metrics);

  /**
   * @brief Prints performance report and optimization insights.
   * @param compiler_coalesced True if compiler analysis reports coalesced memory access.
   * @return None.
   * @sideeffects Writes formatted report and insights to stdout.
   */
  void report(bool compiler_coalesced) const;

 private:
  std::vector<Metrics> metrics_;
};

}  // namespace opengpu::profiler

#endif  // OPENGPU_LAB_PROFILER_PROFILER_H_
