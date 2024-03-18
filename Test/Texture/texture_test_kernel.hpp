
#include "../../Orochi/Orochi.h"


//
// simple code to slightly modify the pixels of an input texture, and output the result into a surface.
// 
extern "C" __global__ void texture_test( 
	oroTextureObject_t texObj, 
	oroSurfaceObject_t surfObj, 
	int width, 
	int height
	)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) 
	{
		uchar4 dataOut;

		float4 dataIn = tex2D<float4>(texObj, x, y);

		dataOut.x = min((int)(dataIn.x*255.0f + 40.0f), 255);
		dataOut.y = max((int)(dataIn.y*255.0f - 40.0f), 0);
		dataOut.z = max((int)(dataIn.z*255.0f - 40.0f), 0);

		dataOut.w = 255;

		surf2Dwrite(dataOut, surfObj, x * sizeof(uchar4), y);
	}

	return;
}
