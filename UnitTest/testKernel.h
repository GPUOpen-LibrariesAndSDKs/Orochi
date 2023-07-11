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

extern "C" __global__ void streamData(float *ptr, const size_t size, 
                              float* output, const float val) 
{ 
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x; 
  size_t n = size / sizeof(float); 
  float accum = 0.0f; 

  for(; tid < n; tid += blockDim.x * gridDim.x) 
  {
  		accum += ptr[tid]; 
  }
   output[threadIdx.x + blockIdx.x * blockDim.x] = accum; 
}
