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

#pragma once
#include <Orochi/Orochi.h>
#include <mutex>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <optional>

#if defined( GNUC )
#include <signal.h>
#endif

template<typename T, typename U>
constexpr void OROASSERT( T&& exp, [[maybe_unused]] U&& placeholder ) noexcept
{
	if( static_cast<bool>( std::forward<T>( exp ) ) != true )
	{

#if defined( _WIN32 )
		__debugbreak();
#elif defined( GNUC )
		raise( SIGTRAP );
#else
		;
#endif
	}
}

class OrochiUtils
{
  public:
	struct int4
	{
		int x, y, z, w;
	};

	OrochiUtils() = default;
	OrochiUtils(const OrochiUtils&) = delete; 
    OrochiUtils& operator=(const OrochiUtils&) = delete;
    OrochiUtils(OrochiUtils&&) = delete; 
    OrochiUtils& operator=(OrochiUtils&&) = delete;
	~OrochiUtils();

	// unload all the modules internally created during functions like getFunctionFromPrecompiledBinary/getFunction
	// good practice to call it just before oroCtxDestroy, just to avoid any potential memory leak.
	void unloadKernelCache();

	oroFunction getFunctionFromPrecompiledBinary( const std::string& path, const std::string& funcName );

	// this function is like 'getFunctionFromPrecompiledBinary' but instead of giving a path to a file, we give the data directly.
	// ( use the script convert_binary_to_array.py to convert the .hipfb to a C-array. )
	oroFunction getFunctionFromPrecompiledBinary_asData( const unsigned char* data, size_t dataSizeInBytes, const std::string& funcName );

	oroFunction getFunctionFromFile( oroDevice device, const char* path, const char* funcName, std::vector<const char*>* opts );
	oroFunction getFunctionFromString( oroDevice device, const char* source, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders, const char** headers, const char** includeNames );
	oroFunction getFunction( oroDevice device, const char* code, const char* path, const char* funcName, std::vector<const char*>* opts, int numHeaders = 0, const char** headers = 0, const char** includeNames = 0, oroModule* loadedModule = 0 );

	static bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	static void getData( oroDevice device, const char* code, const char* path, std::vector<const char*>* opts, std::vector<char>& dst );
	static int getProgram( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, orortcProgram* prog );
	static void getModule( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, oroModule* moduleOut );
	static void launch1D( oroFunction func, int nx, const void** args, int wgSize = 64, unsigned int sharedMemBytes = 0, oroStream stream = 0 );
	static void launch2D( oroFunction func, int nx, int ny, const void** args, int wgSizeX = 8, int wgSizeY = 8, unsigned int sharedMemBytes = 0, oroStream stream = 0 );
	

	struct CompressedBuffer {
		const unsigned char* data = nullptr; // compressed data
		size_t size = 0; // size in byte of 'data'
		size_t uncompressedSize = 0; // size of byte of the uncompressed data.
	};
	struct RawBuffer {
		const unsigned char* data = nullptr;
		size_t size = 0;
	};
	static void HandlePrecompiled(std::vector<unsigned char>& out, const CompressedBuffer& buffer);
	static void HandlePrecompiled(std::vector<unsigned char>& out, const RawBuffer& buffer);
	static void HandlePrecompiled(std::vector<unsigned char>& out, const unsigned char* rawData, size_t rawData_sizeByte, std::optional<size_t> uncompressed_sizeByte=std::nullopt);

	template<typename T>
	static void malloc( T*& ptr, size_t n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void mallocManaged( T*& ptr, size_t n, oroManagedMemoryAttachFlags flags )
	{
		oroError e = oroMallocManaged( (oroDeviceptr*)&ptr, sizeof( T ) * n, flags );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void free( T* ptr )
	{
		oroFree( (oroDeviceptr)ptr );
	}

	static void memset( void* ptr, int val, size_t n )
	{
		oroError e = oroMemset( ptr, val, n );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void memsetAsync( void* ptr, int val, size_t n, oroStream stream )
	{
		oroError e = oroMemsetD8Async( ptr, val, n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoD( T* dst, const T* src, size_t n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoH( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoH( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoD( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyHtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyHtoDAsync( (oroDeviceptr)dst, (void*)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoHAsync( (void*)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	template<typename T>
	static void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoDAsync( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		OROASSERT( e == oroSuccess, 0 );
	}

	static void waitForCompletion( oroStream stream = 0 )
	{
		auto e = oroStreamSynchronize( stream );
		OROASSERT( e == oroSuccess, 0 );
	}

  public:
	std::string m_cacheDirectory = "./cache/";
	std::recursive_mutex m_mutex;

	struct FunctionModule {
		oroFunction function;
		oroModule module;
	};

	std::unordered_map<std::string, FunctionModule> m_kernelMap;
};

class OroStopwatch
{
  public:
	OroStopwatch( oroStream stream ) 
	{ 
		m_stream = stream;
		oroEventCreateWithFlags( &m_start, oroEventDefault );
		oroEventCreateWithFlags( &m_stop, oroEventDefault );
	}
	~OroStopwatch() 
	{
		oroEventDestroy( m_start );
		oroEventDestroy( m_stop );
	}

	void start() { oroEventRecord( m_start, m_stream ); }
	void stop() { oroEventRecord( m_stop, m_stream ); }

	float getMs() 
	{ 
		oroEventSynchronize( m_stop );
		float ms = 0;
		oroEventElapsedTime( &ms, m_start, m_stop );
		return ms;
	}

  public:
	oroStream m_stream;
	oroEvent m_start;
	oroEvent m_stop;
};