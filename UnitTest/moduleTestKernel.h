extern __device__ void setInfo( int* x ); 

extern "C" __global__ void testKernel( int *x )
{ 
	setInfo(x);
}
