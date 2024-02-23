# simpleD3D12 - Simple D3D12 Oro Interop

## Description

Based on Direct3D12 interoperability with CUDA (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D12). The program creates a sinewave in DX12 vertex buffer which is created using CUDA/hip kernels. DX12 and CUDA/hip synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA/AMD GPU is required on Windows10 or higher OS.


## Involved functions

oroWaitExternalSemaphoresAsync, oroExternalMemoryGetMappedBuffer, oroImportExternalSemaphore, oroSignalExternalSemaphoresAsync, oroDestroyExternalMemory, oroImportExternalMemory, oroDestroyExternalSemaphore

## Build and Run

The sln file is provided. 

## Licenses

The Direct3D12 interoperability with CUDA (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D12) is based on this license:  
  
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.  
  
Redistribution and use in source and binary forms, with or without  
modification, are permitted provided that the following conditions  
are met:  
 * Redistributions of source code must retain the above copyright  
   notice, this list of conditions and the following disclaimer.  
 * Redistributions in binary form must reproduce the above copyright  
   notice, this list of conditions and the following disclaimer in the  
   documentation and/or other materials provided with the distribution.  
 * Neither the name of NVIDIA CORPORATION nor the names of its  
   contributors may be used to endorse or promote products derived  
   from this software without specific prior written permission.  
  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY  
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE  
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR  
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR  
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR  
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY  
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT  
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
  
For additional information on the license terms, see the CUDA EULA at  
https://docs.nvidia.com/cuda/eula/index.html  
  
  
  
  
  
It's also using source code from https://github.com/microsoft/DirectX-Graphics-Samples  
  
The MIT License (MIT)  
Copyright (c) 2015 Microsoft  
  
Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  
  
The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  
  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.  

