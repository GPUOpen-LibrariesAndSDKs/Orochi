#include <hip/hip_runtime.h>

extern "C" __global__ void writeValue( int* value ) { *value = 42; }
