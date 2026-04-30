cmake_minimum_required(VERSION 3.20)

# filename: CMakeLists.txt
# purpose: Root build configuration for OpenGPU Lab.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

project(opengpu_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

enable_testing()

add_subdirectory(runtime)
add_subdirectory(verilator)
add_subdirectory(backends)
add_subdirectory(scheduler)
add_subdirectory(compiler)
add_subdirectory(profiler)
add_subdirectory(tests)
