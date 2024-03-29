//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Orochi/GpuMemory.h>
#include <Test/Common.h>
#include <fstream>
#include "../../UnitTest/demoErrorCodes.h"

// use a third-party library half.hpp to use the fp16 half dataype on the host side
#include "half.hpp"
using __half = half_float::half;

int main( int argc, char** argv )
{
	bool testErrorFlag = false;
	OrochiUtils o;

	// Initialize Orochi
	// only ORO_API_HIP because this Demo currently only works on HIP
	if ( oroInitialize( ( oroApi )( ORO_API_HIP ), 0 ) != 0 ) 
	{ 
		printf( "Unable to initialize Orochi. Please check your HIP installation or create an issue at our github for assistance.\n" );
		return OROCHI_TEST_RETCODE__ERROR;
	}

	oroError e;
	e = oroInit( 0 );
	ERROR_CHECK( e );

	oroDevice device;
	// Get the device at index 0 (choose the index corresponding to your RDNA3 GPU in case you have multiple GPUs)
	e = oroDeviceGet( &device, 0 );
	ERROR_CHECK( e );

	char name[128];
	e = oroDeviceGetName( name, 128, device );
	ERROR_CHECK( e );

	oroDeviceProp props;
	e = oroGetDeviceProperties( &props, device );
	ERROR_CHECK( e );
	printf( "executing on %s (%s)\n", props.name, props.gcnArchName );

	oroCtx ctx;
	e = oroCtxCreate( &ctx, 0, device );
	ERROR_CHECK( e );
	oroCtxSetCurrent( ctx );


	oroFunction function = o.getFunctionFromFile(device, "../Test/WMMA/wmma_test_kernel.h", "wmma_matmul", nullptr);


	__half a[16 * 16] = {};
	__half b[16 * 16] = {};
	__half c[16 * 16] = {};
	__half *a_gpu, *b_gpu, *c_gpu;
	oroMalloc((oroDeviceptr*)&a_gpu, 16*16 * sizeof(__half));
	oroMalloc((oroDeviceptr*)&b_gpu, 16*16 * sizeof(__half));
	oroMalloc((oroDeviceptr*)&c_gpu, 16*16 * sizeof(__half));

	// fill in some data into matrices A and B
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			a[i * 16 + j] = (__half)1.f;
			b[i * 16 + j] = (__half)1.f;
		}
	}

	oroMemcpyHtoD((oroDeviceptr)a_gpu, (void*)a, (16*16) * sizeof(__half));
	oroMemcpyHtoD((oroDeviceptr)b_gpu, (void*)b, (16*16) * sizeof(__half));
	oroMemcpyHtoD((oroDeviceptr)c_gpu, (void*)c, (16*16) * sizeof(__half));

	const void* args[] = {&a_gpu, &b_gpu, &c_gpu};
	e = oroModuleLaunchKernel( function, 1, 1, 1, 32, 1, 1, 0, 0, (void**)args, 0 ); 
	oroDeviceSynchronize();

	oroMemcpyDtoH(c, (oroDeviceptr)c_gpu, (16 * 16) * sizeof(__half));

	oroFree((oroDeviceptr)a_gpu);
	oroFree((oroDeviceptr)b_gpu);
	oroFree((oroDeviceptr)c_gpu);

	printf( "Output matrix:\n" );
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			printf("%f ", (float)c[i * 16 + j]);
		}
		printf("\n");
	} 
	printf( "Done!\n" );
	e = oroCtxDestroy( ctx );



	if ( testErrorFlag )
		return OROCHI_TEST_RETCODE__ERROR;
	return OROCHI_TEST_RETCODE__SUCCESS;
}
