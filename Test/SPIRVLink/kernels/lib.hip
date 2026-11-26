#include "common.h"
__device__ Vec3 libVec( Vec3 v ) { return Vec3{ v.x * 2.0f, v.y * 2.0f, v.z * 2.0f }; }
__device__ int  libInt( int x ) { return x * 3 + 1; }
