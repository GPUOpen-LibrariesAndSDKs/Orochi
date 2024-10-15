#define GLEW_STATIC
#include <contrib/glew/include/glew/glew.h>
#include <contrib/glfw/include/GLFW/glfw3.h>

#include <iostream>
#include <assert.h>

//#define USE_CUDA 1

#if !defined(USE_CUDA)
#include <Orochi/Orochi.h>
inline void checkError( const oroError result )
{
	if( result != oroSuccess )
	{
		const char* errorNamePtr = oroGetErrorName( result );
		const std::string_view errorName{ ( errorNamePtr != nullptr ) ? errorNamePtr : "NoErrorName" };
		std::cerr << "oroError '" << errorName << "': ";
		const char* errorDescriptionPtr = nullptr;
		oroGetErrorString( result, &errorDescriptionPtr );
		const std::string_view errorDescription{ ( errorDescriptionPtr != nullptr ) ? errorDescriptionPtr : "No description." };
		std::cerr << errorDescription << '\n' << std::flush;
		std::abort();
	}
}

inline void checkError( const orortcResult result )
{
	if( result != ORORTC_SUCCESS )
	{
		const std::string_view errorDescription = orortcGetErrorString( result );
		std::cerr << "orortcError: " << errorDescription << '\n' << std::flush;
		std::abort();
	}
}
#else

#pragma comment( lib, "cuda.lib" )
#pragma comment( lib, "cudart.lib" )
//#include <contrib/cuew/include/cuew.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

inline void checkError( const CUresult result )
{
	if( result != CUDA_SUCCESS )
	{
		assert( 0 );
	}
}

inline void checkError( const cudaError_t result )
{
	if( result != CUDA_SUCCESS )
	{
		assert( 0 );
	}
}

inline void checkError( const nvrtcResult result )
{
	if( result != CUDA_SUCCESS )
	{
		assert( 0 );
	}
}
#endif

#define ERROR_CHECK( e ) checkError( e )

#if !defined( USE_CUDA )
int main( int argc, char** argv ) 
{ 
	GLFWwindow* window;
	if( !glfwInit() ) return 0;

	glfwWindowHint( GLFW_RED_BITS, 32 );
	glfwWindowHint( GLFW_GREEN_BITS, 32 );
	glfwWindowHint( GLFW_BLUE_BITS, 32 );

	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );

	window = glfwCreateWindow( 1280, 720, "orochiOglInterop", NULL, NULL );
	if( !window )
	{
		glfwTerminate();
		return 0;
	}
	glfwMakeContextCurrent( window );

	if( glewInit() != GLEW_OK )
	{
		glfwTerminate();
		return 0;
	}

	oroDevice m_device = 0;
	oroCtx m_ctx = nullptr;
	oroStream m_stream = nullptr;

	{
		const int deviceIndex = 0;
		oroApi api = (oroApi)( ORO_API_CUDA | ORO_API_HIP );
		int a = oroInitialize( api, 0 );
		assert( a == 0 );

		ERROR_CHECK( oroInit( 0 ) );
		ERROR_CHECK( oroDeviceGet( &m_device, deviceIndex ) );
		ERROR_CHECK( oroCtxCreate( &m_ctx, 0, m_device ) );
		ERROR_CHECK( oroCtxSetCurrent( m_ctx ) );
		ERROR_CHECK( oroStreamCreate( &m_stream ) );
	}

	{
		const uint32_t imageSize = 64u;
		GLuint texture = 0;
		oroGraphicsResource_t oroResource = nullptr;
		oroArray_t interopArray = nullptr;
		oroTextureObject_t interopTexture = nullptr;

		// Work around for AMD driver crash when calling `oroGLRegister*` functions
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		if( isAmd )
		{
			uint32_t deviceCount = 16;
			int glDevices[16];
			ERROR_CHECK( hipGLGetDevices( &deviceCount, glDevices, deviceCount, hipGLDeviceListAll ) );
		}

		// Create texture object
		glGenTextures( 1, &texture );
		glBindTexture( GL_TEXTURE_2D, texture );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, imageSize, imageSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );

		// Register framebuffer attachment that will be passed to Oro
		{
			ERROR_CHECK( oroGraphicsGLRegisterImage( &oroResource, texture, GL_TEXTURE_2D, oroGraphicsRegisterFlagsReadOnly ) );
			oroDeviceSynchronize();
			ERROR_CHECK( oroGetLastError() );
			// Map buffer objects to get device pointers
			ERROR_CHECK( oroGraphicsMapResources( 1, &oroResource, 0 ) );
			oroDeviceSynchronize();
			ERROR_CHECK( oroGetLastError() );
			ERROR_CHECK( oroGraphicsSubResourceGetMappedArray( &interopArray, oroResource, 0, 0 ) );
			oroDeviceSynchronize();
			ERROR_CHECK( oroGetLastError() );

			// Create texture interop object to be passed to kernel
			{
				oroChannelFormatDesc desc;
				ERROR_CHECK( oroGetChannelDesc( &desc, interopArray ) );
				oroDeviceSynchronize();
				ERROR_CHECK( oroGetLastError() );

				oroResourceDesc texRes;
				memset( &texRes, 0, sizeof( oroResourceDesc ) );

				texRes.resType = oroResourceTypeArray;
				texRes.res.array.array = interopArray;

				oroTextureDesc texDescr;
				memset( &texDescr, 0, sizeof( oroTextureDesc ) );

				texDescr.normalizedCoords = false;
				texDescr.filterMode = oroFilterModePoint;
				texDescr.addressMode[0] = oroAddressModeWrap;
				texDescr.readMode = oroReadModeElementType;

				ERROR_CHECK( oroCreateTextureObject( &interopTexture, &texRes, &texDescr, NULL ) );
			}
		}

		ERROR_CHECK( oroGraphicsUnmapResources( 1, &oroResource, 0 ) );
		ERROR_CHECK( oroDestroyTextureObject( interopTexture ) );
		ERROR_CHECK( oroGraphicsUnregisterResource( oroResource ) );

		glDeleteTextures( 1, &texture );
	}

	ERROR_CHECK( oroStreamDestroy( m_stream ) );
	ERROR_CHECK( oroCtxDestroy( m_ctx ) );


	std::cout << "Success!\n";
	return 0;
}
#else
int main( int argc, char** argv )
{
	GLFWwindow* window;
	if( !glfwInit() ) return 0;

	glfwWindowHint( GLFW_RED_BITS, 32 );
	glfwWindowHint( GLFW_GREEN_BITS, 32 );
	glfwWindowHint( GLFW_BLUE_BITS, 32 );

	glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );

	window = glfwCreateWindow( 1280, 720, "orochiOglInterop", NULL, NULL );
	if( !window )
	{
		glfwTerminate();
		return 0;
	}
	glfwMakeContextCurrent( window );

	if( glewInit() != GLEW_OK )
	{
		glfwTerminate();
		return 0;
	}

	CUdevice m_device = 0;
	CUcontext  m_ctx = nullptr;
	CUstream m_stream = nullptr;

	{
		const int deviceIndex = 0;
//		oroApi api = (oroApi)( ORO_API_CUDA | ORO_API_HIP );
//		int a = oroInitialize( api, 0 );
//		assert( a == 0 );

		ERROR_CHECK( cuInit( 0 ) );
		ERROR_CHECK( cuDeviceGet( &m_device, deviceIndex ) );
		ERROR_CHECK( cuCtxCreate( &m_ctx, 0, m_device ) );
		ERROR_CHECK( cuCtxSetCurrent( m_ctx ) );
		ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
	}

	{
		const uint32_t imageSize = 64u;
		GLuint texture = 0;
		cudaGraphicsResource* oroResource;
		cudaArray_t interopArray = nullptr;
		cudaTextureObject_t interopTexture = 0;
/*
		// Work around for AMD driver crash when calling `oroGLRegister*` functions
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		if( isAmd )
		{
			uint32_t deviceCount = 16;
			int glDevices[16];
			ERROR_CHECK( hipGLGetDevices( &deviceCount, glDevices, deviceCount, hipGLDeviceListAll ) );
		}
*/
		// Create texture object
		glGenTextures( 1, &texture );
		glBindTexture( GL_TEXTURE_2D, texture );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, imageSize, imageSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
		glBindTexture( GL_TEXTURE_2D, 0 );

		// Register framebuffer attachment that will be passed to Oro
		{
			ERROR_CHECK( cudaGraphicsGLRegisterImage( &oroResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly ) );
			cudaDeviceSynchronize();
			ERROR_CHECK( cudaGetLastError() );
			// Map buffer objects to get device pointers
			ERROR_CHECK( cudaGraphicsMapResources( 1, &oroResource, 0 ) );
			cudaDeviceSynchronize();
			ERROR_CHECK( cudaGetLastError() );
			ERROR_CHECK( cudaGraphicsSubResourceGetMappedArray( &interopArray, oroResource, 0, 0 ) );
			cudaDeviceSynchronize();
			ERROR_CHECK( cudaGetLastError() );

			// Create texture interop object to be passed to kernel
			{
				cudaChannelFormatDesc desc;
				ERROR_CHECK( cudaGetChannelDesc( &desc, interopArray ) );
				cudaDeviceSynchronize();
				ERROR_CHECK( cudaGetLastError() );

				cudaResourceDesc texRes;
				memset( &texRes, 0, sizeof( cudaResourceDesc ) );

				texRes.resType = cudaResourceTypeArray;
				texRes.res.array.array = interopArray;

				cudaTextureDesc texDescr;
				memset( &texDescr, 0, sizeof( cudaTextureDesc ) );

				texDescr.normalizedCoords = false;
				texDescr.filterMode = cudaFilterModePoint;
				texDescr.addressMode[0] = cudaAddressModeWrap;
				texDescr.readMode = cudaReadModeElementType;

				ERROR_CHECK( cudaCreateTextureObject( &interopTexture, &texRes, &texDescr, NULL ) );
			}

		}

//		ERROR_CHECK( oroGraphicsUnmapResources( 1, &oroResource, 0 ) );
//		ERROR_CHECK( oroDestroyTextureObject( interopTexture ) );
//		ERROR_CHECK( oroGraphicsUnregisterResource( oroResource ) );

		glDeleteTextures( 1, &texture );
	}

	ERROR_CHECK( cuStreamDestroy( m_stream ) );
	ERROR_CHECK( cuCtxDestroy( m_ctx ) );

	std::cout << "Success!\n";
	return 0;
}
#endif