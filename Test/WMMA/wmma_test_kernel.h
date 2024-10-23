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

// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

// Use frag_type as an alias of the internal clang vector type of 16 fp16 values

//#define ENABLE_TEST 1

#if defined(ENABLE_TEST)
#define WMMA_DATA_WIDTH 8
typedef _Float16 frag_type __attribute__( ( ext_vector_type( 8 ) ) );
#else
#define WMMA_DATA_WIDTH 16
typedef _Float16 frag_type __attribute__( ( ext_vector_type( 16 ) ) );
#endif

extern "C" __global__ void wmma_matmul( __half* a, __half* b, __half* c )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int lIdx = threadIdx.x;

	// a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
	// a_frag will store one column of the 16x16 matrix tile
	// b_frag will store one row of the 16x16 matrix tile
	frag_type a_frag;
	frag_type b_frag;
	// initialize c fragment to 0
	frag_type c_frag = {};

	// lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA3
	const int lane = lIdx % 16;
	const int laneGroup = lIdx / 16;
#if defined( ENABLE_TEST )
	for( int ele = 0; ele < WMMA_DATA_WIDTH; ++ele )
	{
		b_frag[ele] = b[16 * (ele+laneGroup * WMMA_DATA_WIDTH) + lane];
	}

	for( int ele = 0; ele < WMMA_DATA_WIDTH; ++ele )
	{
		a_frag[ele] = a[16 * lane + ele+laneGroup * WMMA_DATA_WIDTH];
	}
#else
	for( int ele = 0; ele < WMMA_DATA_WIDTH; ++ele )
	{
		b_frag[ele] = b[16 * ele + lane];
	}

	for( int ele = 0; ele < WMMA_DATA_WIDTH; ++ele )
	{
		a_frag[ele] = a[16 * lane + ele];
	}
#endif
	// call the WMMA compiler intrinsic 
	// more details available in the RDNA3 ISA guide - https://developer.amd.com/wp-content/resources/RDNA3_Shader_ISA_December2022.pdf
	// the last parameter is called "OPSEL" which decides which half of the VGPRs of c_frag the results are stored into
	// this will only compile on RDNA3
#if defined( ENABLE_TEST )
	c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12( a_frag, b_frag, c_frag );
#else
	c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32( a_frag, b_frag, c_frag, false );
#endif
#if defined( ENABLE_TEST )
	for( int ele = 0; ele < WMMA_DATA_WIDTH; ++ele )
	{
		c[16 * ( ele + laneGroup * WMMA_DATA_WIDTH ) + lane] = c_frag[ele];
	}
#else
	for( int ele = 0; ele < 8; ++ele )
	{
		const int r = ele * 2 + ( lIdx / 16 );
		// store results from unpacked c_frag output
		c[16 * r + lane] = c_frag[ele * 2];
		// if OPSEL was set to "true", the line above would instead be
		// c[16 * r + lane] = c_frag[ele*2 + 1];
	}
#endif
}
