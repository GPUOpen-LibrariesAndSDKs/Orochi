To compile the two kernels on navi21:

`hipcc --offload-arch=gfx1030 moduleTestFunc.cpp -c -fgpu-rdc --cuda-device-only`
`hipcc --offload-arch=gfx1030 moduleTestKernel.cpp -c -fgpu-rdc --cuda-device-only`
