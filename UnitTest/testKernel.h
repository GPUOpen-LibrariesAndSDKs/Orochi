extern "C" __global__ void testKernel()
{
	int a = threadIdx.x;
	printf( "	thread %d running\n", a );
}
