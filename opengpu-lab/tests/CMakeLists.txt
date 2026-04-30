# filename: CMakeLists.txt
# purpose: Builds and registers test binaries.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_executable(test_matmul_cpu test_matmul_cpu.cpp)
target_include_directories(test_matmul_cpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../backends/cpu/kernels
)
target_link_libraries(test_matmul_cpu PRIVATE backend_cpu)
add_test(NAME test_matmul_cpu COMMAND test_matmul_cpu)

add_executable(test_matmul_cuda test_matmul_cuda.cpp)
target_include_directories(test_matmul_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../backends/cpu/kernels
)
target_link_libraries(test_matmul_cuda PRIVATE backend_cpu backend_cuda runtime)
add_test(NAME test_matmul_cuda COMMAND test_matmul_cuda)

add_executable(test_dispatcher test_dispatcher.cpp)
target_link_libraries(test_dispatcher PRIVATE backend_cpu backend_cuda runtime)
add_test(NAME test_dispatcher COMMAND test_dispatcher)

add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE scheduler runtime)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_rtl_sim test_rtl_sim.cpp)
target_link_libraries(test_rtl_sim PRIVATE backend_cpu backend_rtl_sim runtime)
add_test(NAME test_rtl_sim COMMAND test_rtl_sim)

add_executable(test_rtl_validation test_rtl_validation.cpp)
target_link_libraries(test_rtl_validation PRIVATE verilator_harness)
add_test(NAME test_rtl_validation COMMAND test_rtl_validation)

add_executable(test_profiler test_profiler.cpp)
target_include_directories(test_profiler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../backends/cpu/kernels
)
target_link_libraries(test_profiler PRIVATE
  backend_cpu
  backend_cuda
  backend_rtl_sim
  scheduler
  profiler
  runtime
)
add_test(NAME test_profiler COMMAND test_profiler)

add_executable(test_compiler test_compiler.cpp)
target_link_libraries(test_compiler PRIVATE
  compiler
  profiler
)
target_compile_definitions(test_compiler PRIVATE
  CUDA_KERNEL_PATH="${CMAKE_SOURCE_DIR}/backends/cuda/kernels/matmul.cu"
)
add_test(NAME test_compiler COMMAND test_compiler)

add_executable(test_golden test_golden.cpp)
target_include_directories(test_golden PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../backends/cpu/kernels
)
target_link_libraries(test_golden PRIVATE backend_cpu)

add_executable(test_scheduler_stub test_scheduler_stub.cpp)
target_link_libraries(test_scheduler_stub PRIVATE scheduler)
