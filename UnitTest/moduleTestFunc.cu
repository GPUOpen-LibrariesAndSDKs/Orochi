 __device__ void setInfo( int *x )
{
	int tid = threadIdx.x;
	atomicAdd( x, tid );
}
