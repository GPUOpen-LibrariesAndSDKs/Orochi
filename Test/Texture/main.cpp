//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//


#include <cstring> // std::memset
#include <Orochi/GpuMemory.h>
#include <Orochi/Orochi.h>
#include <Test/Common.h>
#include <iostream>
#include "contrib/stb/stb_image_write.h"
#include "contrib/stb/stb_image.h"
#include "../../UnitTest/demoErrorCodes.h"



struct IMG_PIXEL_UCHAR4 
{
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
};


void writeImageToPNG(const IMG_PIXEL_UCHAR4* image, int width, int height, const char* filename) 
{
	// Prepare an array for the RGB data
	unsigned char *rgbImage = (unsigned char *)malloc(width * height * 3);

	// Convert uchar4 to RGB format
	for (size_t i = 0; i < width * height; ++i) {
		rgbImage[i * 3] = image[i].x; // R
		rgbImage[i * 3 + 1] = image[i].y; // G
		rgbImage[i * 3 + 2] = image[i].z; // B
	}

	// Write to PNG
	stbi_write_png(filename, width, height, 3, rgbImage, width * 3);

	free(rgbImage);
}


int main()
{
	bool testErrorFlag = false;
	OrochiUtils o;

	oroApi api = ( oroApi )( ORO_API_CUDA | ORO_API_HIP );
	if( oroInitialize( api, 0 ) != 0 )
	{
		std::cerr << "Unable to initialize Orochi. Please check your HIP installation or create an issue at our github for assistance.\n";
		return OROCHI_TEST_RETCODE__ERROR;
	}

	oroError e{};
	oroDevice device{};
	
	ERROR_CHECK( oroInit( 0 ) );

	// Get the device at index 0
	ERROR_CHECK( oroDeviceGet( &device, 0 ) );

	static constexpr auto name_size = 128;
	char name[name_size];
	ERROR_CHECK( oroDeviceGetName( name, name_size, device ) );

	oroDeviceProp props{};
	ERROR_CHECK( oroGetDeviceProperties( &props, device ) );
	printf( "executing on %s (%s)\n", props.name, props.gcnArchName );

	oroCtx ctx{};
	ERROR_CHECK( oroCtxCreate( &ctx, 0, device ) );
	ERROR_CHECK( oroCtxSetCurrent( ctx ) );


	std::vector<const char*> opts;
	opts.push_back( "-I../" );
	oroFunction function = o.getFunctionFromFile(device, "../Test/Texture/texture_test_kernel.hpp", "texture_test", &opts);


	static constexpr auto grid_resolution = 256;
	static constexpr auto num_features = 4;

	Oro::GpuMemory<float> grid_data( grid_resolution * grid_resolution * num_features );

	int stbi_dimX = 0;
	int stbi_dimY = 0;
	int stbi_comp = 0;
	stbi_uc* imgInStbi = stbi_load("../Test/resources/nature.png", &stbi_dimX, &stbi_dimY, &stbi_comp, 0);

	if ( !imgInStbi )
	{
		printf("ERROR: can't open image.\n");
		return OROCHI_TEST_RETCODE__ERROR;
	}

	if (   stbi_dimX != grid_resolution
		|| stbi_dimY != grid_resolution
		|| stbi_comp != 4
		)
	{
		printf("ERROR: TODO improve management of image input\n");
		return OROCHI_TEST_RETCODE__ERROR;
	}

	std::vector<float> test_data_temp_img_cpu;
	test_data_temp_img_cpu.resize(grid_resolution * grid_resolution * num_features);
	for(int i=0; i<grid_resolution*grid_resolution; i++)
	{
		test_data_temp_img_cpu[i*4+0] = (float)imgInStbi[i*4+0] / 255.0f;
		test_data_temp_img_cpu[i*4+1] = (float)imgInStbi[i*4+1] / 255.0f;
		test_data_temp_img_cpu[i*4+2] = (float)imgInStbi[i*4+2] / 255.0f;
		test_data_temp_img_cpu[i*4+3] = 1.0;
	}

	grid_data.copyFromHost( std::data( test_data_temp_img_cpu ), std::size( test_data_temp_img_cpu ) );

	oroError_t status = oroSuccess;

	oroArray_Format format = ORO_AD_FORMAT_FLOAT;

	// Resource Desc
	ORO_RESOURCE_DESC resDesc;
	std::memset( &resDesc, 0, sizeof( resDesc ) );
	resDesc.resType = ORO_RESOURCE_TYPE_PITCH2D;
	resDesc.res.pitch2D.devPtr = reinterpret_cast<oroDeviceptr_t>( grid_data.ptr() );
	resDesc.res.pitch2D.format = format;
	resDesc.res.pitch2D.numChannels = num_features;
	resDesc.res.pitch2D.width = grid_resolution;
	resDesc.res.pitch2D.height = grid_resolution;
	resDesc.res.pitch2D.pitchInBytes = grid_resolution * sizeof( float ) * num_features;

	OROaddress_mode address_mode = ORO_TR_ADDRESS_MODE_WRAP;
	OROfilter_mode filter_mode = ORO_TR_FILTER_MODE_POINT;

	ORO_TEXTURE_DESC texDesc;
	std::memset( &texDesc, 0, sizeof( texDesc ) );
	texDesc.addressMode[0] = address_mode;
	texDesc.addressMode[1] = address_mode;
	texDesc.addressMode[2] = address_mode;
	texDesc.filterMode = filter_mode;
	texDesc.flags = ORO_TRSF_READ_AS_INTEGER;

	oroTextureObject_t textureObject{};
	ERROR_CHECK( oroTexObjectCreate( &textureObject, &resDesc, &texDesc, nullptr ) );

	int width = grid_resolution;
	int height = grid_resolution;

	oroSurfaceObject_t surfObj = nullptr;
	oroArray_t oroArray = nullptr;
	{
		
		oroChannelFormatDesc channelDesc;
		channelDesc.x = 8;
		channelDesc.y = 8;
		channelDesc.z = 8;
		channelDesc.w = 8;
		channelDesc.f = oroChannelFormatKindUnsigned;

		status = oroMallocArray(&oroArray, &channelDesc, width, height, oroArrayDefault);
		if (status != ORO_SUCCESS) 
		{
			std::cerr << "Failed to allocate array" << std::endl;
			return OROCHI_TEST_RETCODE__ERROR;
		}


		// Specify surface resource description
		oroResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = oroResourceTypeArray;
		resDesc.res.array.array = oroArray;

		// Create the surface object
		status = oroCreateSurfaceObject(&surfObj, &resDesc);
		if (status != ORO_SUCCESS) 
		{
			oroArrayDestroy(oroArray); // Cleanup array if surface creation fails
			e = oroCtxDestroy( ctx ); 
			ERROR_CHECK( e );
			return OROCHI_TEST_RETCODE__ERROR;
		}

	}


	oroStream stream;
	ERROR_CHECK( oroStreamCreate( &stream ) );

	const int blockDimX = 16;
	const int blockDimY = 16;

	const int gridDimX = (width + blockDimX - 1) / blockDimX ;
	const int gridDimY = (height + blockDimY - 1) / blockDimY; 


	// Launch the kernel
	const void* args[] = { &textureObject, &surfObj, &grid_resolution, &grid_resolution };
	ERROR_CHECK( oroModuleLaunchKernel(function, gridDimX, gridDimY, 1, blockDimX,blockDimY,1,  0 , stream, (void**)args, nullptr ) );
	ERROR_CHECK( oroDeviceSynchronize() );


	IMG_PIXEL_UCHAR4* hostImage = (IMG_PIXEL_UCHAR4*)malloc(width * height * sizeof(IMG_PIXEL_UCHAR4));
	if (hostImage == nullptr) 
	{
		std::cerr << "Failed to allocate host image memory" << std::endl;
		return OROCHI_TEST_RETCODE__ERROR;
	}


	size_t dpitch = width * sizeof(IMG_PIXEL_UCHAR4); // Destination pitch
	status = oroMemcpy2DFromArray(hostImage, dpitch, oroArray, 0, 0, width * sizeof(IMG_PIXEL_UCHAR4), height, oroMemcpyDeviceToHost);
	if (status != ORO_SUCCESS) 
	{
		std::cerr << "Failed to copy data from array to host" << std::endl;
	}


	// Optional Code - specific for Unit Tests:
	// do the same compute, but on CPU side, to check the results
	{
		for(int i=0; i<grid_resolution*grid_resolution; i++)
		{
			if ( 
			   hostImage[i].x != std::min(imgInStbi[i*4+0] + 40, 255)
			|| hostImage[i].y != std::max(imgInStbi[i*4+1] - 40, 0)
			|| hostImage[i].z != std::max(imgInStbi[i*4+2] - 40, 0)
			|| hostImage[i].w != imgInStbi[i*4+3]
				)
			{
				testErrorFlag = true;
				std::cerr << "ERROR: The output image seems wrong compared to reference." << std::endl;
				break;
			}
		}
	}


	std::string outFile = "texture_out.png";
	writeImageToPNG(hostImage, width, height, outFile.c_str());
	std::cout<< "file " + outFile + " has been created.\n";


	stbi_image_free(imgInStbi); imgInStbi=nullptr;
	ERROR_CHECK( oroStreamDestroy( stream ) );
	ERROR_CHECK( oroDestroySurfaceObject(surfObj)) ;  surfObj=nullptr;
	ERROR_CHECK( oroArrayDestroy(oroArray)) ;  oroArray=nullptr;
	o.unloadKernelCache();
	ERROR_CHECK( oroCtxDestroy( ctx ) );

	if ( testErrorFlag )
		return OROCHI_TEST_RETCODE__ERROR;
	return OROCHI_TEST_RETCODE__SUCCESS;
}
