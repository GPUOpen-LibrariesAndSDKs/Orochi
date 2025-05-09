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

#include <Orochi/OrochiUtils.h>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#else
#include <errno.h>
#include <locale>
#include <sys/stat.h>
#endif

#ifdef ORO_LINK_ZSTD
#include <contrib/zstd/lib/zstd.h>
#endif

inline std::wstring utf8_to_wstring( const std::string& str )
{
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> myconv;
	std::wstring out1 = myconv.from_bytes( str );
	return out1;
}

class FileStat
{
#if defined( _WIN32 )
  public:
	FileStat( const char* filePath )
	{
		m_file = 0;
		std::wstring filePathW = utf8_to_wstring( filePath );
		m_file = CreateFileW( filePathW.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0 );
		if( m_file == INVALID_HANDLE_VALUE )
		{
			DWORD errorCode;
			errorCode = GetLastError();
			switch( errorCode )
			{
			case ERROR_FILE_NOT_FOUND:
			{
#ifdef _DEBUG
				printf( "File not found %s\n", filePath );
#endif
				break;
			}
			case ERROR_PATH_NOT_FOUND:
			{
#ifdef _DEBUG
				printf( "File path not found %s\n", filePath );
#endif
				break;
			}
			default:
			{
				printf( "Failed reading file with errorCode = %d\n", static_cast<int>( errorCode ) );
				printf( "%s\n", filePath );
			}
			}
		}
	}
	~FileStat()
	{
		if( m_file != INVALID_HANDLE_VALUE ) CloseHandle( m_file );
	}

	bool found() const { return ( m_file != INVALID_HANDLE_VALUE ); }

	unsigned long long getTime()
	{
		if( m_file == INVALID_HANDLE_VALUE ) return 0;

		unsigned long long t = 0;
		FILETIME exeTime;
		if( GetFileTime( m_file, NULL, NULL, &exeTime ) == 0 )
		{
		}
		else
		{
			unsigned long long u = exeTime.dwHighDateTime;
			t = ( u << 32 ) | exeTime.dwLowDateTime;
		}
		return t;
	}

  private:
	HANDLE m_file;
#else
  public:
	FileStat( const char* filePath ) { m_file = filePath; }

	bool found() const
	{
		struct stat binaryStat;
		bool e = stat( m_file.c_str(), &binaryStat );
		return e == 0;
	}

	unsigned long long getTime()
	{
		struct stat binaryStat;
		bool e = stat( m_file.c_str(), &binaryStat );
		if( e != 0 ) return 0;
		unsigned long long t = binaryStat.st_mtime;
		return t;
	}

  private:
	std::string m_file;
#endif
};

struct OrochiUtilsImpl
{
	static bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes )
	{
		std::fstream f( path );
		if( f.is_open() )
		{
			size_t sizeFile;
			f.seekg( 0, std::fstream::end );
			size_t size = sizeFile = (size_t)f.tellg();
			f.seekg( 0, std::fstream::beg );
			if( includes )
			{
				sourceCode.clear();
				std::string line;
				char buf[512];
				while( std::getline( f, line ) )
				{
					if( strstr( line.c_str(), "#include" ) != 0 )
					{
						const char* a = strstr( line.c_str(), "<" );
						const char* b = strstr( line.c_str(), ">" );
						int n = b - a - 1;
						memcpy( buf, a + 1, n );
						buf[n] = '\0';
						includes->push_back( buf );
						sourceCode += line + '\n';
					}
					else
					{
						sourceCode += line + '\n';
					}
				}
			}
			else
			{
				sourceCode.resize( size, ' ' );
				f.read( &sourceCode[0], size );
			}
			f.close();
			return true;
		}
		return false;
	}

	static void getCacheFileName( oroDevice device, const char* moduleName, const char* functionName, const char* options, std::string& binFileName, const std::string& cacheDirectory )
	{
		auto hashBin = []( const char* s, const size_t size )
		{
			unsigned int hash = 0;

			for( unsigned int i = 0; i < size; ++i )
			{
				hash += *s++;
				hash += ( hash << 10 );
				hash ^= ( hash >> 6 );
			}

			hash += ( hash << 3 );
			hash ^= ( hash >> 11 );
			hash += ( hash << 15 );

			return hash;
		};

		auto hashString = [&]( const char* ss, const size_t size, char buf[9] )
		{
			const unsigned int hash = hashBin( ss, size );

			sprintf( buf, "%08x", hash );
		};

		auto strip = []( const char* name, const char* pattern )
		{
			size_t const patlen = strlen( pattern );
			size_t patcnt = 0;
			const char* oriptr;
			const char* patloc;
			// find how many times the pattern occurs in the original string
			for( oriptr = name; ( patloc = strstr( oriptr, pattern ) ); oriptr = patloc + patlen )
			{
				patcnt++;
			}
			return oriptr;
		};

		oroDeviceProp props;
		oroGetDeviceProperties( &props, device );

		int v = 0;
		oroDriverGetVersion( &v );
		std::string deviceName = props.name;
		std::string driverVersion = std::to_string( v );
		char optionHash[9] = "0x0";

		if( moduleName && options )
		{
			std::string tmp = moduleName;
			tmp += options;

			hashString( tmp.c_str(), strlen( tmp.c_str() ), optionHash );
		}

		char moduleHash[9] = "0x0";
		const char* strippedModuleName = strip( moduleName, "\\" );
		strippedModuleName = strip( strippedModuleName, "/" );
		hashString( strippedModuleName, strlen( strippedModuleName ), moduleHash );

		using namespace std::string_literals;

		deviceName = deviceName.substr( 0, deviceName.find( ":" ) );
		binFileName = cacheDirectory + "/"s + moduleHash + "-"s + optionHash + ".v."s + deviceName + "."s + driverVersion + "_"s + std::to_string( 8 * sizeof( void* ) ) + ".bin"s;
		return;
	}
	static bool isFileUpToDate( const char* binaryFileName, const char* srcFileName )
	{
		FileStat b( binaryFileName );

		if( !b.found() ) return false;

		FileStat s( srcFileName );

		if( !s.found() )
		{
			//	todo. compare with exe time
			return true;
		}

		if( s.getTime() < b.getTime() ) return true;

		return false;
	}

	static bool createDirectory( const char* cacheDirName )
	{
#if defined( _WIN32 )
		std::wstring cacheDirNameW = utf8_to_wstring( cacheDirName );
		bool error = CreateDirectoryW( cacheDirNameW.c_str(), 0 );
		if( error == false && GetLastError() != ERROR_ALREADY_EXISTS )
		{
			printf( "Cache folder path not found!\n" );
			return false;
		}
		return true;
#else
		int error = mkdir( cacheDirName, 0775 );
		if( error == -1 && errno != EEXIST )
		{
			printf( "Cache folder path not found!\n" );
			return false;
		}
		return true;
#endif
	}
	static std::string getCheckSumFileName( const std::string& binaryName )
	{
		const std::string dst = binaryName + ".check";
		return dst;
	}
	static inline long long checksum( const char* data, long long size )
	{
		unsigned int hash = 0;

		for( unsigned int i = 0; i < size; ++i )
		{
			hash += *data++;
			hash += ( hash << 10 );
			hash ^= ( hash >> 6 );
		}

		hash += ( hash << 3 );
		hash ^= ( hash >> 11 );
		hash += ( hash << 15 );

		return hash;
	}

	static int loadCacheFileToBinary( const std::string& cacheName, std::vector<char>& binaryOut )
	{
		long long checksumValue = 0;
		{
			const std::string csFileName = getCheckSumFileName( cacheName );
#if defined( _WIN32 )
			std::wstring csFileNameW = utf8_to_wstring( csFileName );
			FILE* csfile = _wfopen( csFileNameW.c_str(), L"rb" );
#else
			FILE* csfile = fopen( csFileName.c_str(), "rb" );
#endif
			if( csfile )
			{
				fread( &checksumValue, sizeof( long long ), 1, csfile );
				fclose( csfile );
			}
		}

		if( checksumValue == 0 ) return 0;

#if defined( _WIN32 )
		std::wstring binaryFileNameW = utf8_to_wstring( cacheName );
		FILE* file = _wfopen( binaryFileNameW.c_str(), L"rb" );
#else
		FILE* file = fopen( cacheName.c_str(), "rb" );
#endif
		if( file )
		{
			fseek( file, 0L, SEEK_END );
			size_t binarySize = ftell( file );
			rewind( file );

			binaryOut.resize( binarySize );
			size_t dummy = fread( const_cast<char*>( binaryOut.data() ), sizeof( char ), binarySize, file );
			fclose( file );

			long long s = checksum( binaryOut.data(), binarySize );
			if( s != checksumValue )
			{
				printf( "checksum doesn't match %llx : %llx\n", s, checksumValue );
				return 0;
			}
		}
		return 0;
	}

	static int cacheBinaryToFile( std::vector<char> binary, const std::string& cacheName )
	{
		const size_t binarySize = binary.size();
		{
#ifdef WIN32
			std::wstring binaryFileNameW = utf8_to_wstring( cacheName );
			FILE* file = _wfopen( binaryFileNameW.c_str(), L"wb" );
#else
			FILE* file = fopen( cacheName.c_str(), "wb" );
#endif

			if( file )
			{
#ifdef _DEBUG
				printf( "Cached file created %s\n", cacheName.c_str() );
#endif
				fwrite( binary.data(), sizeof( char ), binarySize, file );
				fclose( file );
			}
		}

		long long s = checksum( const_cast<char*>( binary.data() ), binarySize );
		const std::string filename = getCheckSumFileName( cacheName );

		{
#ifdef WIN32
			std::wstring filenameW = utf8_to_wstring( filename );
			FILE* file = _wfopen( filenameW.c_str(), L"wb" );
#else
			FILE* file = fopen( filename.c_str(), "wb" );
#endif

			if( file )
			{
				fwrite( &s, sizeof( long long ), 1, file );
				fclose( file );
			}
		}
		return 0;
	}

	static std::string getCacheName( const std::string& path, const std::string& kernelname, std::vector<const char*>* opts ) noexcept
	{
		std::string tmp_name = path + kernelname;

		if (opts == nullptr)
			return tmp_name;

		for( std::string s : *opts )
		{
			if( s.size() > 1 && s[0] == '-' && s[1] == 'I' ) continue;
			tmp_name += s;
		}
		return tmp_name;
	}

	static std::string getCacheName( const std::string& path, const std::string& kernelname ) noexcept { return path + kernelname; }
};

void OrochiUtils::unloadKernelCache() 
{
	for ( auto& instance : m_kernelMap ) 
	{
		oroError e = oroModuleUnload( instance.second.module );
		OROASSERT( e == oroSuccess, 0 );
	}
	m_kernelMap.clear();
	return;
}

OrochiUtils::~OrochiUtils() 
{
	// it's safer to not call unloadKernelCache automatically in the destructor ( better to have a leak than manipulating bad pointers )
	// Just inform the developer.
	if ( m_kernelMap.size() > 0 )
		printf("Warning: OrochiUtils::unloadKernelCache should be called for good practice.\n");
}

// setup a base option list used in OrochiUtils 
void SetupCompileOptions(
	oroDevice device, 
	const std::vector<const char*>* optsIn, 
	std::string* optionalArchitectureTarget, // if this string is given, it must stay alive until it's used into orortcCompileProgram. ( otherwise its "c_str()" becomes an invalid pointer )
	std::vector<const char*>& opts
	)
{
	opts.push_back( "-std=c++17" );

	if( optionalArchitectureTarget && oroGetCurAPI( 0 ) == ORO_API_HIP )
	{
		oroDeviceProp props;
		::memset(&props,0,sizeof(props));
		oroGetDeviceProperties( &props, device );
		if ( props.gcnArchName && props.gcnArchName[0] != '\0' )
		{
			*optionalArchitectureTarget = "--gpu-architecture=";
			*optionalArchitectureTarget += props.gcnArchName;
			opts.push_back( optionalArchitectureTarget->c_str() );
		}
	}

	if( optsIn )
	{
		for( int i = 0; i < optsIn->size(); i++ )
			opts.push_back( ( *optsIn )[i] );
	}
}

// base program compilation, used in several components of OrochiUtils
// returns 0 if success.
// if fails, returns a non-zero value
int CreateAndCompileProgram(const char* code, const char* programName, std::vector<const char*>& opts, const char* nameExpression, 
							int numHeaders, const char** headers, const char** includeNames,
							orortcProgram* prog)
{
	orortcResult e;
	e = orortcCreateProgram( prog, code, programName, numHeaders, headers, includeNames );

	if ( nameExpression )
		e = orortcAddNameExpression( *prog, nameExpression );

	e = orortcCompileProgram( *prog, static_cast<int>( opts.size() ), opts.data() );
	if( e != ORORTC_SUCCESS )
	{
		std::cout << "ERROR: orortcCompileProgram failed with log:\n";
		size_t logSize = 0;
		orortcGetProgramLogSize( *prog, &logSize );
		if( logSize )
		{
			std::string log( logSize, '\0' );
			orortcGetProgramLog( *prog, &log[0] );
			std::cout << log << '\n';
		}
		else
		{
			std::cout << "<NO LOG>\n";
		}

		// Even if orortcCompileProgram failed, let's return 'success' because there's a known issue where this function returns ORORTC_ERROR_COMPILATION even though the program might have been compiled correctly.
		//    ( example with Orochi Unit Test:  TEST_F( OroTestBase, link )   )
		// The failure check is done at a higher level in functions like 'orortcGetBitcodeSize', so we won't miss any actual errors.
		return 0;
	}
	return 0; // success code
}

bool OrochiUtils::readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes ) 
{
	return OrochiUtilsImpl::readSourceCode( path, sourceCode, includes ); 
}

// returns nullptr if failed
oroFunction OrochiUtils::getFunctionFromFile( oroDevice device, const char* path, const char* funcName, std::vector<const char*>* optsIn )
{
	std::lock_guard<std::recursive_mutex> lock( m_mutex );

	const std::string cacheName = OrochiUtilsImpl::getCacheName( path, funcName, optsIn );
	if( m_kernelMap.find( cacheName.c_str() ) != m_kernelMap.end() )
	{
		return m_kernelMap[cacheName].function;
	}

	std::string source;
	if( !OrochiUtilsImpl::readSourceCode( path, source, 0 ) ) 
	{
		printf("WARNING: getFunctionFromFile of file %s failed.\n", path);
		return 0;
	}

	oroModule module = nullptr;
	oroFunction f = getFunction( device, source.c_str(), path, funcName, optsIn, 0, nullptr, nullptr, &module );

	if ( f )
	{
		m_kernelMap[cacheName].function = f;
		m_kernelMap[cacheName].module = module;
	}

	return f;
}

oroFunction OrochiUtils::getFunctionFromString( oroDevice device, const char* source, const char* path, const char* funcName, std::vector<const char*>* optsIn, int numHeaders, const char** headers, const char** includeNames )
{
	std::lock_guard<std::recursive_mutex> lock( m_mutex );

	const std::string cacheName = OrochiUtilsImpl::getCacheName( path, funcName, optsIn );
	if( m_kernelMap.find( cacheName.c_str() ) != m_kernelMap.end() )
	{
		return m_kernelMap[cacheName].function;
	}

	oroModule module;

	oroFunction f = getFunction( device, source, path, funcName, optsIn, numHeaders, headers, includeNames, &module );

	if ( f )
	{
		m_kernelMap[cacheName].function = f;
		m_kernelMap[cacheName].module = module;
	}
	
	return f;
}

oroFunction OrochiUtils::getFunctionFromPrecompiledBinary_asData( const unsigned char* precompData, size_t dataSizeInBytes, const std::string& funcName )
{
	std::lock_guard<std::recursive_mutex> lock( m_mutex );

	const std::string cacheName = OrochiUtilsImpl::getCacheName( "___BAKED_BIN___", funcName );
	if( m_kernelMap.find( cacheName.c_str() ) != m_kernelMap.end() )
	{
		return m_kernelMap[cacheName].function;
	}

	oroModule module = nullptr;
	oroError e = oroModuleLoadData( &module, precompData );
	if ( e != oroSuccess )
	{
		// add some verbose info to help debugging missing data
		printf("oroModuleLoadData FAILED (error = %d) loading baked precomp data: %s\n", e, funcName.c_str());
		return nullptr;
	}

	oroFunction functionOut{};
	e = oroModuleGetFunction( &functionOut, module, funcName.c_str() );
	if ( e != oroSuccess )
	{
		// add some verbose info to help debugging missing data
		printf("oroModuleGetFunction FAILED (error = %d) loading baked precomp data: %s\n", e, funcName.c_str());
		return nullptr;
	}
	OROASSERT( e == oroSuccess, 0 );

	m_kernelMap[cacheName].function = functionOut;
	m_kernelMap[cacheName].module = module;

	return functionOut;
}

oroFunction OrochiUtils::getFunctionFromPrecompiledBinary( const std::string& path, const std::string& funcName )
{
	std::lock_guard<std::recursive_mutex> lock( m_mutex );

	const std::string cacheName = OrochiUtilsImpl::getCacheName( path, funcName );
	if( m_kernelMap.find( cacheName.c_str() ) != m_kernelMap.end() )
	{
		return m_kernelMap[cacheName].function;
	}

	std::ifstream instream( path, std::ios::in | std::ios::binary );
	if ( !instream || !instream.is_open() )
	{
		printf("OrochiUtils::getFunctionFromPrecompiledBinary FAILED to open file: %s\n", path.c_str());
		return nullptr;
	}

	std::vector<char> binary( ( std::istreambuf_iterator<char>( instream ) ), std::istreambuf_iterator<char>() );

	oroModule module = nullptr;
	oroFunction functionOut{};
	oroError e = oroModuleLoadData( &module, binary.data() );
	if ( e != oroSuccess )
	{
		// add some verbose info to help debugging missing file
		printf("oroModuleLoadData FAILED (error = %d) loading file: %s\n", e, path.c_str());
		return nullptr;
	}
	OROASSERT( e == oroSuccess, 0 );

	e = oroModuleGetFunction( &functionOut, module, funcName.c_str() );
	OROASSERT( e == oroSuccess, 0 );

	m_kernelMap[cacheName].function = functionOut;
	m_kernelMap[cacheName].module = module;

	return functionOut;
}

// returns nullptr if failed
oroFunction OrochiUtils::getFunction( oroDevice device, const char* code, const char* path, const char* funcName, std::vector<const char*>* optsIn, int numHeaders, const char** headers, const char** includeNames, oroModule* loadedModule)
{
	std::lock_guard<std::recursive_mutex> lock( m_mutex );

	std::vector<const char*> opts;
	SetupCompileOptions(device, optsIn, nullptr, opts);

	oroFunction function;
	std::vector<char> codec;

	std::string cacheFile;
	{
		std::string o;
		for( int i = 0; i < opts.size(); i++ )
			o.append( opts[i] );
		OrochiUtilsImpl::getCacheFileName( device, path, funcName, o.c_str(), cacheFile, m_cacheDirectory );
	}
	if( OrochiUtilsImpl::isFileUpToDate( cacheFile.c_str(), path ) )
	{
		// load cache
		OrochiUtilsImpl::loadCacheFileToBinary( cacheFile, codec );
	}
	else
	{
		orortcProgram prog = nullptr;
		int createProgramErrorCode = CreateAndCompileProgram(code, funcName, opts, nullptr, numHeaders, headers, includeNames, &prog);

		// if CreateAndCompileProgram failed
		if ( createProgramErrorCode != 0 )
		{
			if ( prog )
				orortcDestroyProgram( &prog );
			return nullptr;
		}

		size_t codeSize;
		orortcResult e;
		e = orortcGetCodeSize( prog, &codeSize );
		OROASSERT( e == ORORTC_SUCCESS, 0 );

		codec.resize( codeSize );
		e = orortcGetCode( prog, codec.data() );
		OROASSERT( e == ORORTC_SUCCESS, 0 );
		e = orortcDestroyProgram( &prog );
		OROASSERT( e == ORORTC_SUCCESS, 0 );

		// store cache
		OrochiUtilsImpl::createDirectory( m_cacheDirectory.c_str() );
		OrochiUtilsImpl::cacheBinaryToFile( codec, cacheFile );
	}
	oroModule module;
	oroError ee = oroModuleLoadData( &module, codec.data() );
	OROASSERT( ee == oroSuccess, 0 );
	ee = oroModuleGetFunction( &function, module, funcName );
	OROASSERT( ee == oroSuccess, 0 );

	if ( loadedModule ) 
	{
		*loadedModule = module;
	}

	return function;
}

void OrochiUtils::getData( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, std::vector<char>& dst )
{
	orortcProgram prog = nullptr;
	int getProgErrorCode = getProgram( device, code, path, optsIn, nullptr, &prog );
	if ( getProgErrorCode != 0 )
	{
		printf("WARNING: getProgram failed.\n");
		if ( prog )
			orortcDestroyProgram( &prog );
		return;
	}

	orortcResult e = ORORTC_SUCCESS;
	size_t codeSize = 0;
	e = orortcGetBitcodeSize( prog, &codeSize );
	if ( e != ORORTC_SUCCESS || codeSize == 0 )
	{
		printf("WARNING: orortcGetBitcodeSize failed.\n");
		if ( prog )
			orortcDestroyProgram( &prog );
		return;
	}

	std::vector<char>& codec = dst;
	codec.resize( codeSize );
	e = orortcGetBitcode( prog, codec.data() );
	e = orortcDestroyProgram( &prog );
	
	return;
}

// returns 0 if SUCCEED
// return non-zero value if FAILED
int OrochiUtils::getProgram( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, orortcProgram* prog )
{
	std::vector<const char*> opts;
	std::string architectureTarget;
	SetupCompileOptions(device, optsIn, &architectureTarget, opts);
	int createProgramErrorCode = CreateAndCompileProgram(code, path, opts, funcName, 0, nullptr, nullptr, prog);
	return createProgramErrorCode;
}

void OrochiUtils::getModule( oroDevice device, const char* code, const char* path, std::vector<const char*>* optsIn, const char* funcName, oroModule* moduleOut ) 
{ 
	orortcProgram prog = nullptr;
	int getProgErrorCode = getProgram( device, code, path, optsIn, funcName, &prog );
	if ( getProgErrorCode != 0 )
	{
		printf("WARNING: getProgram failed.\n");
		if ( prog )
			orortcDestroyProgram( &prog );
		return;
	}

	size_t codeSize = 0;
	orortcResult e = ORORTC_SUCCESS;
	std::vector<char> codec;
	e = orortcGetCodeSize( prog, &codeSize );
	if ( e != ORORTC_SUCCESS || codeSize == 0 )
	{
		printf("WARNING: orortcGetCodeSize failed.\n");
		if ( prog )
			orortcDestroyProgram( &prog );
		return;
	}

	codec.resize( codeSize );
	e = orortcGetCode( prog, codec.data() );
	OROASSERT( e == ORORTC_SUCCESS, 0 );
	e = orortcDestroyProgram( &prog );
	OROASSERT( e == ORORTC_SUCCESS, 0 );

	oroError ee = oroModuleLoadData( moduleOut, codec.data() );
	OROASSERT( ee == oroSuccess, 0 );
	return;
}

void OrochiUtils::launch1D( oroFunction func, int nx, const void** args, int wgSize, unsigned int sharedMemBytes, oroStream stream ) 
{
	int4 tpb = { wgSize, 1, 0 };
	int4 nb = { ( nx + tpb.x - 1 ) / tpb.x, 1, 0 };
	oroError e = oroModuleLaunchKernel( func, nb.x, nb.y, 1, tpb.x, tpb.y, 1, sharedMemBytes, stream, (void**)args, 0 );
	OROASSERT( e == oroSuccess, 0 );
}

void OrochiUtils::launch2D( oroFunction func, int nx, int ny, const void** args, int wgSizeX, int wgSizeY, unsigned int sharedMemBytes, oroStream stream )
{
	int4 tpb = { wgSizeX, wgSizeY, 0 };
	int4 nb = { ( nx + tpb.x - 1 ) / tpb.x, ( ny + tpb.y - 1 ) / tpb.y, 0 };
	oroError e = oroModuleLaunchKernel( func, nb.x, nb.y, 1, tpb.x, tpb.y, 1, sharedMemBytes, stream, (void**)args, 0 );
	OROASSERT( e == oroSuccess, 0 );
}

void OrochiUtils::HandlePrecompiled(std::vector<unsigned char>& out, const CompressedBuffer& buffer)
{
	#ifdef ORO_LINK_ZSTD
		out.assign(buffer.uncompressedSize,0);

		size_t decompressedSize = ZSTD_decompress(    
			out.data(), // final uncompressed buffer
			out.size(), // final size
			buffer.data, // compressed buffer
			buffer.size // compressed buffer - size
			);
		
		if ( decompressedSize != buffer.uncompressedSize )
			throw std::runtime_error( "ERROR: ZSTD_decompress FAILED." );
	#else
		throw std::runtime_error( "ERROR: ZSTD is not part of this build." );
	#endif
	return;
}


void OrochiUtils::HandlePrecompiled(std::vector<unsigned char>& out, const RawBuffer& buffer)
{
	out = std::vector<unsigned char>(buffer.data, buffer.data + buffer.size );
	return;
}


void OrochiUtils::HandlePrecompiled(std::vector<unsigned char>& out, const unsigned char* rawData, size_t rawData_sizeByte, std::optional<size_t> uncompressed_sizeByte)
{
	if (uncompressed_sizeByte.has_value()) {
		// if the input buffer is compressed :
		CompressedBuffer buffer{ rawData, rawData_sizeByte, uncompressed_sizeByte.value() };
		HandlePrecompiled(out, buffer );
	} else {
		// if the input buffer is not compressed
		RawBuffer buffer{ rawData, rawData_sizeByte };
		HandlePrecompiled(out, buffer );
	}
}


