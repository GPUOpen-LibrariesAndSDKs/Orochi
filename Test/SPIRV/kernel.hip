#include <hip/hip_runtime.h>

extern "C" __global__ void transform( int* values, unsigned count )
{
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < count ) values[i] = 3 * values[i] + 7;
}
