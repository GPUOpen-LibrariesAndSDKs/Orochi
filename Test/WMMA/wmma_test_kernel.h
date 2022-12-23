// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <hip/hip_runtime.h>

using namespace std;

// Use half16 as an alias of the internal clang vector type of 16 fp16 values
typedef _Float16 half16 __attribute__( ( ext_vector_type( 16 ) ) );

extern "C" __global__ void wmma_matmul( __half* a, __half* b, __half* c )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int lIdx = threadIdx.x;

	// a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
	// a_frag will store one column of the 16x16 matrix tile
	// b_frag will store one row of the 16x16 matrix tile
	half16 a_frag;
	half16 b_frag;
	// initialize c fragment to 0
	half16 c_frag = {};

	// lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA3
	const int lane = lIdx % 16;

	for( int ele = 0; ele < 16; ++ele )
	{
		b_frag[ele] = b[16 * ele + lane];
	}

	for( int ele = 0; ele < 16; ++ele )
	{
		a_frag[ele] = a[16 * lane + ele];
	}

	// call the WMMA compiler intrinsic 
	// more details available in the RDNA3 ISA guide - https://developer.amd.com/wp-content/resources/RDNA3_Shader_ISA_December2022.pdf
	// the last parameter is called "OPSEL" which decides which decides which half of the VGPRs of c_frag the results are stored into
	// this will only compile on RDNA3
	c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32( a_frag, b_frag, c_frag, false );

	for( int ele = 0; ele < 8; ++ele )
	{
		const int r = ele * 2 + ( lIdx / 16 );
		// store results from unpacked c_frag output
		c[16 * r + lane] = c_frag[ele * 2];
		// if OPSEL was set to "true", the line above would instead be
		// c[16 * r + lane] = c_frag[ele*2 + 1];
	}
}
