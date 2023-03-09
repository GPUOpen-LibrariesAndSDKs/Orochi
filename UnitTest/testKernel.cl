__kernel void testKernel( __global int* a )
{
	int tid = get_global_id( 0 );
	atomic_add(a, tid);
}
