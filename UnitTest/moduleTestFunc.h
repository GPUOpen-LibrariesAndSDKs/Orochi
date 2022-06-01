 __device__ void printInfo()
{
	int a = threadIdx.x;
	printf( "	thread %d running\n", a );
}
