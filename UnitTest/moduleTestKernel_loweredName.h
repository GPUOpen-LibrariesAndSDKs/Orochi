extern __device__ void setInfo( int *x ); 

template<int XYZ>
__global__ void testKernel( int *x )
{ 
	setInfo(x);
}
