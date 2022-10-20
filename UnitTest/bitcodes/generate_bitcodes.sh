hipcc --cuda-device-only --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1033 --offload-arch=gfx1034 --offload-arch=gfx1035 --offload-arch=gfx1036 --offload-arch=gfx1010 --offload-arch=gfx1011 --offload-arch=gfx1012 --offload-arch=gfx1013 --offload-arch=gfx900 --offload-arch=gfx906 -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm ../moduleTestKernel.cpp

hipcc --cuda-device-only --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1033 --offload-arch=gfx1034 --offload-arch=gfx1035 --offload-arch=gfx1036 --offload-arch=gfx1010 --offload-arch=gfx1011 --offload-arch=gfx1012 --offload-arch=gfx1013 --offload-arch=gfx900 --offload-arch=gfx906 -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm ../moduleTestFunc.cpp
hipcc --offload-arch=gfx1030 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1031 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1032 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1033 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1034 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1035 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1036 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1010 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1011 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1012 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1013 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx900 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx906 ../moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only

hipcc --offload-arch=gfx1030 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1031 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1032 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1033 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1034 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1035 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1036 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1010 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1011 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1012 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx1013 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx900 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
hipcc --offload-arch=gfx906 ../moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only
