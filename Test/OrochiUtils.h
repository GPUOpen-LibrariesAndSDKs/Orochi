#pragma once
#include <Orochi/Orochi.h>
#include <vector>

#define OROASSERT(x, y) if(!(x)) {__debugbreak();}

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	static oroFunction getFunctionFromFile( const char* path, const char* funcName, std::vector<const char*>* opts );
	static oroFunction getFunction( const char* code, const char* path, const char* funcName, std::vector<const char*>* opts );

	static void launch1D( oroFunction func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0 );

	template<typename T>
	static void malloc( T*& ptr, int n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void free( void* ptr ) { oroFree( (oroDeviceptr)ptr ); }

	static void memset( void* ptr, int val, int n ) { oroMemset( (oroDeviceptr)ptr, val, n ); }

	template<typename T>
	static void copyHtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoH( dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static
	void waitForCompletion()
	{
		auto e = oroDeviceSynchronize();
		OROASSERT( e == oroSuccess, 0 );
	}
};
