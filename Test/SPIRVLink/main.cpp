#include <Orochi/Orochi.h>
#include <UnitTest/demoErrorCodes.h>

#if defined( _WIN32 )
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Test for linking AMDGCN SPIR-V modules at run time with hipLink* (no arguments needed).
//
// `userKernel` (kernels/user.hip) calls `libVec` (kernels/lib.hip), a function that returns a 12-byte struct by
// value, and writes 2 10 40. The project's pre-build step compiles the kernels with the HIP SDK compiler into
// <executable dir>/SPIRVLink/, and two cases are run:
//   1. lib.bundle + user.bundle linked at run time                    <- the case under test
//   2. all.bundle, both units compiled as one module (kernels/all.hip) <- control
// Exit code 100 if both cases produce the expected values, 99 otherwise.
// If arguments are given, they are the module list of a single custom case instead.
//
// Orochi does not wrap hipLink* yet, so the four entry points are resolved from the HIP runtime that hipew
// already loaded.

typedef struct ihipLinkState_t* hipLinkState_t;
constexpr int hipJitInputSpirv = 103;

typedef hipError_t ( *PFN_hipLinkCreate )( unsigned int, void*, void**, hipLinkState_t* );
typedef hipError_t ( *PFN_hipLinkAddData )( hipLinkState_t, int, void*, size_t, const char*, unsigned int, void*, void** );
typedef hipError_t ( *PFN_hipLinkComplete )( hipLinkState_t, void**, size_t* );
typedef hipError_t ( *PFN_hipLinkDestroy )( hipLinkState_t );

static PFN_hipLinkCreate   linkCreate;
static PFN_hipLinkAddData  linkAddData;
static PFN_hipLinkComplete linkComplete;
static PFN_hipLinkDestroy  linkDestroy;

static bool check( oroError status, const char* call )
{
	if( status == oroSuccess ) return true;
	const char* message = nullptr;
	oroGetErrorString( status, &message );
	std::fprintf( stderr, "  %s failed (%d): %s\n", call, static_cast<int>( status ), message ? message : "unknown error" );
	return false;
}

#define CHECK( call ) if( !check( call, #call ) ) return false
#define CHECK_HIP( call ) if( !check( static_cast<oroError>( call ), #call ) ) return false

static bool readFile( const std::string& path, std::vector<char>& data )
{
	std::ifstream input( path, std::ios::binary );
	data.assign( ( std::istreambuf_iterator<char>( input ) ), std::istreambuf_iterator<char>() );
	if( data.empty() ) std::fprintf( stderr, "  cannot read %s\n", path.c_str() );
	return !data.empty();
}

static std::string executableDir()
{
	char exe[4096];
#if defined( _WIN32 )
	const DWORD n = GetModuleFileNameA( nullptr, exe, sizeof( exe ) );
	if( n == 0 || n >= sizeof( exe ) ) return "";
#else
	const ssize_t n = readlink( "/proc/self/exe", exe, sizeof( exe ) - 1 );
	if( n <= 0 ) return "";
#endif
	std::string dir( exe, static_cast<size_t>( n ) );
	dir.erase( dir.find_last_of( "\\/" ) + 1 );
	return dir;
}

static bool runCase( const char* title, const std::vector<std::string>& paths )
{
	std::printf( "%s\n", title );
	std::vector<std::vector<char>> modules( paths.size() );
	for( size_t i = 0; i < paths.size(); ++i )
		if( !readFile( paths[i], modules[i] ) ) return false;

	hipLinkState_t link = nullptr;
	CHECK_HIP( linkCreate( 0, nullptr, nullptr, &link ) );
	for( size_t i = 0; i < modules.size(); ++i )
		CHECK_HIP( linkAddData( link, hipJitInputSpirv, modules[i].data(), modules[i].size(), paths[i].c_str(), 0, nullptr, nullptr ) );
	void*  binary	  = nullptr;
	size_t binarySize = 0;
	CHECK_HIP( linkComplete( link, &binary, &binarySize ) );

	oroModule	module = nullptr;
	oroFunction kernel = nullptr;
	CHECK( oroModuleLoadData( &module, binary ) );
	CHECK( oroModuleGetFunction( &kernel, module, "userKernel" ) );
	CHECK_HIP( linkDestroy( link ) );

	int			 values[3] = { 0, 0, 0 };
	oroDeviceptr deviceValues = nullptr;
	CHECK( oroMalloc( &deviceValues, sizeof( values ) ) );
	CHECK( oroMemsetD8( deviceValues, 0, sizeof( values ) ) );
	const void* args[] = { &deviceValues };
	CHECK( oroModuleLaunchKernel( kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr ) );
	CHECK( oroDeviceSynchronize() );
	CHECK( oroMemcpyDtoH( values, deviceValues, sizeof( values ) ) );
	CHECK( oroFree( deviceValues ) );
	CHECK( oroModuleUnload( module ) );

	const bool ok = values[0] == 2 && values[1] == 10 && values[2] == 40;
	std::printf( "  out = %d %d %d (expected 2 10 40): %s\n", values[0], values[1], values[2], ok ? "PASS" : "FAIL" );
	return ok;
}

int main( int argc, char** argv )
{
	const int initStatus = oroInitialize( ORO_API_HIPDRIVER, 0 );
	if( initStatus != ORO_SUCCESS )
	{
		std::fprintf( stderr, "Cannot load the HIP runtime (%d)\n", initStatus );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	if( !check( oroInit( 0 ), "oroInit" ) ) return OROCHI_TEST_RETCODE__ERROR;
	oroDevice device;
	if( !check( oroDeviceGet( &device, 0 ), "oroDeviceGet" ) ) return OROCHI_TEST_RETCODE__ERROR;
	oroCtx context = nullptr;
	if( !check( oroCtxCreate( &context, 0, device ), "oroCtxCreate" ) ) return OROCHI_TEST_RETCODE__ERROR;
	if( !check( oroCtxSetCurrent( context ), "oroCtxSetCurrent" ) ) return OROCHI_TEST_RETCODE__ERROR;

	// The HIP runtime library is the one that provides hipModuleLoadData to hipew.
	std::string hipPath;
#if defined( _WIN32 )
	HMODULE hip = nullptr;
	if( GetModuleHandleExA(
			GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCSTR>( hipModuleLoadData ),
			&hip ) )
	{
		char path[MAX_PATH];
		if( GetModuleFileNameA( hip, path, MAX_PATH ) ) hipPath = path;
	}
	auto symbol = [&]( const char* name ) { return hip ? reinterpret_cast<void*>( GetProcAddress( hip, name ) ) : nullptr; };
#else
	Dl_info info{};
	void*	hip = nullptr;
	if( dladdr( reinterpret_cast<void*>( hipModuleLoadData ), &info ) && info.dli_fname )
	{
		hipPath = info.dli_fname;
		hip		= dlopen( info.dli_fname, RTLD_NOW | RTLD_NOLOAD );
	}
	auto symbol = [&]( const char* name ) { return hip ? dlsym( hip, name ) : nullptr; };
#endif
	if( !hip )
	{
		std::fprintf( stderr, "Cannot locate the HIP runtime library\n" );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	linkCreate	 = reinterpret_cast<PFN_hipLinkCreate>( symbol( "hipLinkCreate" ) );
	linkAddData	 = reinterpret_cast<PFN_hipLinkAddData>( symbol( "hipLinkAddData" ) );
	linkComplete = reinterpret_cast<PFN_hipLinkComplete>( symbol( "hipLinkComplete" ) );
	linkDestroy	 = reinterpret_cast<PFN_hipLinkDestroy>( symbol( "hipLinkDestroy" ) );
	if( !linkCreate || !linkAddData || !linkComplete || !linkDestroy )
	{
		std::fprintf( stderr, "hipLink* not found in %s (ROCm 6.4 or later is required)\n", hipPath.c_str() );
		return OROCHI_TEST_RETCODE__ERROR;
	}

	bool ok = true;
	if( argc > 1 )
	{
		ok = runCase( "Custom case: modules from the command line linked at run time", std::vector<std::string>( argv + 1, argv + argc ) );
	}
	else
	{
		const std::string dir = executableDir() + "SPIRVLink/";
		if( !std::ifstream( dir + "lib.bundle", std::ios::binary ).good() )
		{
			std::fprintf( stderr, "Cannot find %slib.bundle; the pre-build step compiles it with the HIP SDK compiler\n", dir.c_str() );
			return OROCHI_TEST_RETCODE__ERROR;
		}
		ok = runCase( "Case 1: lib.bundle + user.bundle linked at run time with hipLink*", { dir + "lib.bundle", dir + "user.bundle" } ) && ok;
		ok = runCase( "Case 2 (control): all.bundle, both units compiled as one module", { dir + "all.bundle" } ) && ok;
	}

	check( oroCtxDestroy( context ), "oroCtxDestroy" );
	std::printf( "SPIRVLink: %s\n", ok ? "PASS" : "FAIL" );
	return ok ? OROCHI_TEST_RETCODE__SUCCESS : OROCHI_TEST_RETCODE__ERROR;
}
