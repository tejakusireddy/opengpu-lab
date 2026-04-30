# OpenGPU Lab — Developer shortcuts
# Usage:
#   make analyze   KERNEL=path/to/kernel.cu [N=64]
#   make fix       KERNEL=path/to/kernel.cu [N=64]
#   make benchmark [N=64]

N ?= 64
KERNEL ?= backends/cuda/kernels/matmul.cu

.PHONY: analyze fix benchmark analyze-url fix-url build clean

build:
	@if [ ! -f "./build/tools/gpuopt" ]; then \
		echo "Building gpuopt..."; \
		cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && \
		cmake --build build --target gpuopt > /dev/null 2>&1; \
		echo "Build complete."; \
	fi

analyze: build
	@./build/tools/gpuopt --kernel $(KERNEL) --n $(N)

fix: build
	@./build/tools/gpuopt --kernel $(KERNEL) --n $(N) --fix

analyze-url: build
	@if [ -z "$(URL)" ]; then \
		echo "Usage: make analyze-url URL=https://raw.githubusercontent.com/.../kernel.cu [N=64]"; \
		exit 1; \
	fi
	@echo "Downloading kernel from $(URL)..."
	@curl -s -L -o /tmp/gpuopt_kernel.cu "$(URL)"
	@./build/tools/gpuopt --kernel /tmp/gpuopt_kernel.cu --n $(N)
	@rm -f /tmp/gpuopt_kernel.cu

fix-url: build
	@if [ -z "$(URL)" ]; then \
		echo "Usage: make fix-url URL=https://raw.githubusercontent.com/.../kernel.cu [N=64]"; \
		exit 1; \
	fi
	@echo "Downloading kernel from $(URL)..."
	@curl -s -L -o /tmp/gpuopt_kernel.cu "$(URL)"
	@./build/tools/gpuopt --kernel /tmp/gpuopt_kernel.cu --n $(N) --fix
	@rm -f /tmp/gpuopt_kernel.cu

benchmark:
	@if [ ! -f "./build/tests/test_profiler" ]; then \
		echo "Building benchmarks..."; \
		cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && \
		cmake --build build > /dev/null 2>&1; \
		echo "Build complete."; \
	fi
	@echo "=== OpenGPU Benchmark ==="
	@echo "Kernel: backends/cuda/kernels/matmul.cu"
	@echo "N: $(N)"
	@echo ""
	@./build/tests/test_profiler

clean:
	@rm -rf build
	@echo "Build directory removed."
