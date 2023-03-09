__kernel void testKernel( int x ) 
{ 
	int idx = get_global_id( 0 );
	printf( "%d: %d\n", idx, x );
}

__kernel void testKernel1(__global int* gDst, int x)
{
	int idx = get_global_id(0);
	if (idx == 0)
		gDst[0] = x + idx + 1;

}
