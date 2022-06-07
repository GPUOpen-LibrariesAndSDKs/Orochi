#pragma once
#include <Orochi/Orochi.h>
#include <vector>
#include <unordered_map>
#include <string>

#if defined(_WIN32)
	#define OROASSERT(x, y) if(!(x)) {__debugbreak();}
#else
	#define OROASSERT(x, y) if(!(x)) {;}
#endif

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	OrochiUtils();
	~OrochiUtils();

	oroFunction getFunctionFromFile( oroDevice device, const char* path, const char* funcName, std::vector<const char*>* opts );
	oroFunction getFunctionFromString( oroDevice device, const char* source, const char* path, const char* funcName, std::vector<const char*>* opts, 
		int numHeaders, const char** headers, const char** includeNames );
	oroFunction getFunction( oroDevice device, const char* code, const char* path, const char* funcName, std::vector<const char*>* opts, 
		int numHeaders = 0, const char** headers = 0, const char** includeNames = 0 );

	static void launch1D( oroFunction func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0 );

	template<typename T>
	static void malloc( T*& ptr, int n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void free( T* ptr ) { oroFree( (oroDeviceptr)ptr ); }

	static void memset( void* ptr, int val, int n ) { oroMemset( (oroDeviceptr)ptr, val, n ); }

	template<typename T>
	static void copyHtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoH( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static
	void waitForCompletion()
	{
		auto e = oroDeviceSynchronize();
		OROASSERT( e == oroSuccess, 0 );
	}

public:
	std::string m_cacheDirectory;
	std::unordered_map<std::string, oroFunction> m_kernelMap;
};
