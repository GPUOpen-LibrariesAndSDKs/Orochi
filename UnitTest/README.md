To compile the two kernels on navi21:

`amdclang++ -x hip -O3 -Xclang -fno-cuda-host-device-constexpr --no-offload-new-driver --offload-arch=gfx1030 moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only -emit-llvm`
`amdclang++ -x hip -O3 -Xclang -fno-cuda-host-device-constexpr --no-offload-new-driver --offload-arch=gfx1030 moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only -emit-llvm`

`bitcodes/generate_bitcodes.sh` (`.bat` on Windows) builds the bundled bitcode the tests load for all supported archs. Set `ROCM_PATH` to pick the ROCm install and `OROCHI_OFFLOAD_ARCH` (e.g. `native`) to override the arch list.
