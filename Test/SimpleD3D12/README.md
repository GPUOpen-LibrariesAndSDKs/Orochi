# simpleD3D12 - Simple D3D12 Oro Interop

## Description

Based on Direct3D12 interoperability with CUDA (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D12). The program creates a sinewave in DX12 vertex buffer which is created using CUDA/hip kernels. DX12 and CUDA/hip synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA/AMD GPU is required on Windows10 or higher OS.


## Involved functions

oroWaitExternalSemaphoresAsync, oroExternalMemoryGetMappedBuffer, oroImportExternalSemaphore, oroSignalExternalSemaphoresAsync, oroDestroyExternalMemory, oroImportExternalMemory, oroDestroyExternalSemaphore

## Build and Run

The sln file is provided. 