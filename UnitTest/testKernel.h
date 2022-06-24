extern "C" __global__ void testKernel( int* __restrict__ a )
{
	int tid = threadIdx.x;
	atomicAdd(a, tid);
}
