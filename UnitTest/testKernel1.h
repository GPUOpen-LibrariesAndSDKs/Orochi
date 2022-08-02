#include <UnitTest/testFunc.h>

extern "C" __global__ void testKernel( int* __restrict__ a )
{
	int tid = threadIdx.x;

	theFunc( a, tid );
}
