__device__ void setInfo( int * ); 

extern "C" __global__ void testKernel( int *x )
{ 
	setInfo(x);
}
