#pragma once
#include <hip/hip_runtime.h>
struct Vec3 { float x, y, z; };                 // 12-byte aggregate, like HIP's float3
__device__ Vec3  libVec( Vec3 v );              // returns an aggregate by value, defined in lib.hip
__device__ int   libInt( int x );               // scalar control
