#include <hip/hip_runtime.h>

 __device__ void setInfo( int *x )
{
	int tid = threadIdx.x;
	atomicAdd( x, tid );
}
