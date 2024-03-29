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
#include <Test/Common.h>
#include "../UnitTest/demoErrorCodes.h"

int main(int argc, char** argv )
{
	bool testErrorFlag = false;
	oroApi api = getApiType( argc, argv );

	int a = oroInitialize( api, 0 );
	if( a != 0 )
	{
		printf("initialization failed\n");
		return OROCHI_TEST_RETCODE__ERROR;
	}
	printf( ">> executing on %s\n", ( api == ORO_API_HIP )? "hip":"cuda" );

	printf(">> testing initialization\n");
	ERROR_CHECK(oroInit( 0 ));
	oroDevice device = 0;
	ERROR_CHECK(oroDeviceGet( &device, 0 ));
	oroCtx ctx = nullptr;
	ERROR_CHECK(oroCtxCreate( &ctx, 0, device ));

	printf(">> testing device props\n");
	{
		oroDeviceProp props;
		ERROR_CHECK(oroGetDeviceProperties( &props, device ));
		printf("executing on %s (%s)\n", props.name, props.gcnArchName );
		int v = 0;
		ERROR_CHECK(oroDriverGetVersion( &v ));
		printf("running on driver: %d\n", v);
	}
	printf(">> testing kernel execution\n");
	{
		oroFunction function;
		{
			const char* code = "extern \"C\" __global__ void testKernel( int* __restrict__ a )"
			"{"
				"int tid = threadIdx.x;"
				"atomicAdd( a, tid );"
			"}";
			const char* funcName = "testKernel";
			orortcProgram prog = nullptr;
			
			ERROR_CHECK(orortcCreateProgram( &prog, code, funcName, 0, 0, 0 ));
			std::vector<const char*> opts; 
			opts.push_back( "-I ../" );

			orortcResult e = orortcCompileProgram( prog, opts.size(), opts.data() );
			if( e != ORORTC_SUCCESS )
			{
				std::cout << "orortcCompileProgram FAILED, log:\n";

				size_t logSize = 0;
				orortcGetProgramLogSize(prog, &logSize);
				if (logSize) 
				{
					std::string log(logSize, '\0');
					orortcGetProgramLog(prog, &log[0]);
					std::cout << log << '\n';
				}
				else
				{
					std::cout << "<NO LOG GENERATED>\n";
				}
				return OROCHI_TEST_RETCODE__ERROR;
			}
			size_t codeSize = 0;
			ERROR_CHECK(orortcGetCodeSize(prog, &codeSize));

			std::vector<char> codec(codeSize);
			ERROR_CHECK(orortcGetCode(prog, codec.data()));
			ERROR_CHECK(orortcDestroyProgram(&prog));
			oroModule module;
			ERROR_CHECK(oroModuleLoadData(&module, codec.data()));
			ERROR_CHECK(oroModuleGetFunction(&function, module, funcName));		
		}

		oroStream stream;
		ERROR_CHECK(oroStreamCreate( &stream ));
		
		oroEvent start, stop;
		ERROR_CHECK(oroEventCreateWithFlags( &start, 0 ));
		ERROR_CHECK(oroEventCreateWithFlags( &stop, 0 ));
		ERROR_CHECK(oroEventRecord( start, stream ));

		int a_host = -1;
		int* a_device = nullptr;
		ERROR_CHECK(oroMalloc( (oroDeviceptr*)&a_device, sizeof( int ) ));
		ERROR_CHECK(oroMemset( (oroDeviceptr)a_device, 0, sizeof( int ) ));
		const void* args[] = { &a_device };
		ERROR_CHECK(oroModuleLaunchKernel( function, 1,1,1, 64,1,1, 0, stream, (void**)args, 0 ));

		ERROR_CHECK(oroEventRecord( stop, stream ));

		ERROR_CHECK(oroDeviceSynchronize());

		ERROR_CHECK(oroStreamDestroy( stream ));
		ERROR_CHECK(oroMemcpyDtoH( &a_host, (oroDeviceptr)a_device, sizeof( int ) ));
		printf("a_host (expected 2016): %d\n", a_host);
		ERROR_CHECK(oroFree( (oroDeviceptr)a_device ));

		float milliseconds = 0.0f;
		ERROR_CHECK(oroEventElapsedTime( &milliseconds, start, stop ));
		printf( ">> kernel - %.5f ms\n", milliseconds );
		ERROR_CHECK(oroEventDestroy( start ));
		ERROR_CHECK(oroEventDestroy( stop ));
	}
	printf(">> done\n");

	if ( testErrorFlag )
		return OROCHI_TEST_RETCODE__ERROR;
	return OROCHI_TEST_RETCODE__SUCCESS;
}
