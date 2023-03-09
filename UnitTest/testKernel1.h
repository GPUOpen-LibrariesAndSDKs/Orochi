extern "C" __global__ void testKernel( int x )
{
	int idx = threadIdx.x;
	printf( "%d: %d\n", idx, x );
}

extern "C" __global__ void testKernel1(int* gDst, int x)
{
	int idx = threadIdx.x;
	if (idx == 0)
		gDst[0] = x + idx + 1;

}