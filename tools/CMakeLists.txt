# filename: CMakeLists.txt
# purpose: Builds gpuopt CLI analysis tool.
# phase: CLI v1
# last modified: 2026-04-30

add_executable(gpuopt gpuopt.cpp)
target_link_libraries(gpuopt PRIVATE compiler profiler backend_cuda)
target_include_directories(gpuopt PRIVATE
  ${CMAKE_SOURCE_DIR}/compiler/include
  ${CMAKE_SOURCE_DIR}/profiler/include
  ${CMAKE_SOURCE_DIR}/backends/cuda/include
  ${CMAKE_SOURCE_DIR}/runtime/include
)
