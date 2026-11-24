# SPIR-V on the HIP backend

This example loads an AMDGCN SPIR-V offload bundle with `oroModuleLoadData`,
finds a kernel with `oroModuleGetFunction`, and runs it with `oroModuleLaunchKernel`.
HIP JIT-compiles the bundle for the selected GPU. One thread writes `42`, which
the host copies back and checks.

Requires a ROCm compiler with `amdgcnspirv` support, a HIP runtime supporting
SPIR-V JIT compilation, and a supported AMD GPU.

Build and run on Linux from the repository root (adjust `/opt/rocm` as needed):

```sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64 SPIRV

/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=amdgcnspirv \
    --cuda-device-only -O2 Test/SPIRV/kernel.hip -o dist/kernel.bundle

AMD_COMGR_CACHE=0 ./dist/bin/Release/SPIRV64 dist/kernel.bundle
```

Keep the compiler's offload bundle output enabled: do not add
`--no-gpu-bundle-output`, which produces raw SPIR-V instead of the required bundle.

`AMD_COMGR_CACHE=0` disables the Comgr cache for this run. Success prints
`PASS: value = 42` and exits with **100**; failures exit with **99**, following
Orochi's demo conventions.
