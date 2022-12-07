extern "C" __global__ void testKernel( int* __restrict__ a )
{
	int tid = threadIdx.x;
	atomicAdd(a, tid);
}

// function pointer test

typedef int ( *FuncPointer )( const int idx );

__device__ int testFunc( const int idx )
{
	if( idx == 0 )
		return 7;
	return 0;
}

__device__ FuncPointer gFuncPointer = testFunc;

extern "C" __global__ void testFuncPointerKernel( int* __restrict__ a, FuncPointer* gPointers )
{
	int b = gPointers[0]( threadIdx.x );
	atomicAdd( a, b );
}

