export PATH=$PATH:../../hipsdk/bin
clang++.exe -O3 -fgpu-rdc --hip-link --cuda-device-only --offload-arch=gfx1032 ./moduleTestFunc-hip-amdgcn-amd-amdhsa-gfx1032.bc ./moduleTestKernel-hip-amdgcn-amd-amdhsa-gfx1032.bc -o ./test-gfx1032.fatbin
