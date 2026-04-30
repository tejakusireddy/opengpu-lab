# filename: CMakeLists.txt
# purpose: Builds compiler IR and pass pipeline.
# phase: Phase 8 - Compiler
# last modified: 2026-04-30

add_library(compiler STATIC
  src/ir.cpp
  src/passes.cpp
)

target_include_directories(compiler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(compiler PUBLIC cxx_std_17)
