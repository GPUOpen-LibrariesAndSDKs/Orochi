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
#include "../../UnitTest/demoErrorCodes.h"


int main( int argc, char** argv )
{
	bool testErrorFlag = false;

	int a = oroInitialize( ( oroApi )( ORO_API_CUDA | ORO_API_HIP ), 0 );

	oroError e;
	e = oroInit( 0 );
	int nDevicesTotal;
	e = oroGetDeviceCount( &nDevicesTotal );
	ERROR_CHECK( e );
	int nAMDDevices;
	e = oroGetDeviceCount( &nAMDDevices, ORO_API_HIP );
	ERROR_CHECK( e );
	int nNVIDIADevices;
	e = oroGetDeviceCount( &nNVIDIADevices, ORO_API_CUDA );
	ERROR_CHECK( e );

	printf( "# of devices: %d\n", nDevicesTotal );
	printf( "# of AMD devices: %d\n", nAMDDevices );
	printf( "# of NV devices: %d\n\n", nNVIDIADevices );

	for( int i = 0; i < nDevicesTotal; i++ )
	{
		oroDevice device;
		e = oroDeviceGet( &device, i );
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

		e = oroCtxSetCurrent( ctx );
		ERROR_CHECK( e );

		//try kernel execution
		 oroFunction function;
		{
			const char* code = "extern \"C\" __global__ "
							   "void testKernel()"
							   "{ int a = threadIdx.x; printf(\"	thread %d running\\n\", a); }";
			const char* funcName = "testKernel";
			orortcProgram prog;
			orortcResult e;
			e = orortcCreateProgram( &prog, code, funcName, 0, 0, 0 );
			std::vector<const char*> opts;
			opts.push_back( "-I ../" );

			e = orortcCompileProgram( prog, opts.size(), opts.data() );
			if( e != ORORTC_SUCCESS )
			{
				size_t logSize;
				orortcGetProgramLogSize( prog, &logSize );
				if( logSize )
				{
					std::string log( logSize, '\0' );
					orortcGetProgramLog( prog, &log[0] );
					std::cout << log << '\n';
				};
			}
			size_t codeSize;
			e = orortcGetCodeSize( prog, &codeSize );

			std::vector<char> codec( codeSize );
			e = orortcGetCode( prog, codec.data() );
			e = orortcDestroyProgram( &prog );
			oroModule module;
			oroError ee = oroModuleLoadData( &module, codec.data() );
			ee = oroModuleGetFunction( &function, module, funcName );
		}

		void** args = {};
		oroError e = oroModuleLaunchKernel( function, 1, 1, 1, 32, 1, 1, 0, 0, args, 0 ); 
		oroDeviceSynchronize();

		oroApi api = oroGetCurAPI( 0 );
		printf( "executed on %s\n", api == ORO_API_HIP ? "AMD" : "NVIDIA" );
		e = oroCtxDestroy( ctx );
	}

	if ( testErrorFlag )
		return OROCHI_TEST_RETCODE__ERROR;
	return OROCHI_TEST_RETCODE__SUCCESS;
}
