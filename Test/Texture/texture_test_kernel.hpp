//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//


#include <Orochi/Orochi.h>


//
// simple code to slightly modify the pixels of an input texture, and output the result into a surface.
// 
extern "C" __global__ void texture_test( 
	oroTextureObject_t texObj, 
	oroSurfaceObject_t surfObj, 
	int width, 
	int height
	)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) 
	{
		uchar4 dataOut;

		float4 dataIn = tex2D<float4>(texObj, x, y);

		dataOut.x = min((int)(dataIn.x*255.0f + 40.0f), 255);
		dataOut.y = max((int)(dataIn.y*255.0f - 40.0f), 0);
		dataOut.z = max((int)(dataIn.z*255.0f - 40.0f), 0);
		dataOut.w = 255;

		surf2Dwrite(dataOut, surfObj, x * sizeof(uchar4), y);
	}

	return;
}
