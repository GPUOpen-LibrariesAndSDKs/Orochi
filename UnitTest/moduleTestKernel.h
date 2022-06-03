__device__ void printInfo(); 

extern "C" __global__ void testKernel()
{ 
	printInfo();
}
