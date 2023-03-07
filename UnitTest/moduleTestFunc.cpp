#if !defined( __CUDACC__ )
#include <hip/hip_runtime.h>
#endif

 __device__ void setInfo( int *x )
{
	int tid = threadIdx.x;
	atomicAdd( x, tid );
}
