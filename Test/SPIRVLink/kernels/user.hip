#include "common.h"
extern "C" __global__ void userKernel( int* out )
{
	Vec3 r = libVec( Vec3{ 1.0f, 2.0f, 3.0f } );
	out[0] = static_cast<int>( r.x );        // expect 2
	out[1] = static_cast<int>( r.y + r.z );  // expect 10
	out[2] = libInt( 13 );                   // expect 40
}
