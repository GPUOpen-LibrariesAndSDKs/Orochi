nvcc -x cu -fatbin --device-c -arch=all ../moduleTestFunc.cpp
nvcc -x cu -fatbin --device-c -arch=all ../moduleTestKernel.cpp
