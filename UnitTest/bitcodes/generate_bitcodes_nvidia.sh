nvcc -arch=compute_80 -code="sm_80,sm_86,sm_87" -fatbin --device-c ../moduleTestFunc.cu
nvcc -arch=compute_80 -code="sm_80,sm_86,sm_87" -fatbin --device-c ../moduleTestKernel.cu
nvcc -cubin --device-c -arch=sm_80 ../moduleTestFunc.cu
nvcc -cubin --device-c -arch=sm_80 ../moduleTestKernel.cu
