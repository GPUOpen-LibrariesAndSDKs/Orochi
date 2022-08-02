#if( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#include <UnitTest/testFunc.h>

__device__
void theFunc( int* dst, int idx ) 
{ 
	atomicAdd( dst, idx );
}

#endif
