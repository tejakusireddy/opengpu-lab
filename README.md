# OpenGPU Lab
OpenGPU Lab is a production-oriented GPU systems foundation that establishes a unified runtime, backend interface, scheduler model, hardware simulation hooks, and verification path so CPU, CUDA, and RTL-backed execution can evolve behind one consistent dispatch surface.

## Quick Start
```bash
git clone https://github.com/tejakusireddy/opengpu-lab
cd opengpu-lab
make analyze KERNEL=backends/cuda/kernels/matmul.cu
```

```bash
make analyze KERNEL=your_kernel.cu   # analyze any CUDA file
make fix KERNEL=your_kernel.cu       # detect + auto-fix
make benchmark                        # CPU vs CUDA vs RTL performance
```
