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
//   0. all.bundle loaded with hipModuleLoadData (SPIR-V JIT without any linking) <- does the runtime JIT SPIR-V at all
//   1. lib.spv + user.spv linked at run time with hipLink*                      <- the case under test
//   2. all.spv, both units compiled as one module (kernels/all.hip), via hipLink* <- control for case 1
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

enum class Result { Passed, WrongValues, Error };

#define CHECK( call ) if( !check( call, #call ) ) return Result::Error
#define CHECK_HIP( call ) if( !check( static_cast<oroError>( call ), #call ) ) return Result::Error

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

static void setEnvironmentVariable( const char* name, const char* value )
{
#if defined( _WIN32 )
	SetEnvironmentVariableA( name, value );
	_putenv_s( name, value );
#else
	setenv( name, value, 1 );
#endif
}

static size_t fileSize( const std::string& path )
{
	std::ifstream f( path, std::ios::binary | std::ios::ate );
	return f.good() ? static_cast<size_t>( f.tellg() ) : 0;
}

// Prints the relevant part of what comgr logged from `offset` on, i.e. during the case that just failed:
// the actions it ran, their status and any error or translator message.
static void printComgrLog( const std::string& path, size_t offset )
{
	std::ifstream f( path, std::ios::binary );
	if( !f.good() ) return;
	f.seekg( static_cast<std::streamoff>( offset ) );
	static const char* keys[] = { "ActionKind", "ReturnStatus", "SPIR-V", "SPIRV", "spirv", "rror", "fail", "Fail", "nvalid", "not supported" };
	std::string line;
	std::printf( "  comgr log:\n" );
	while( std::getline( f, line ) )
		for( const char* key : keys )
			if( line.find( key ) != std::string::npos )
			{
				std::printf( "  | %s\n", line.c_str() );
				break;
			}
}

// Runs userKernel from a loaded module and checks its output.
static Result launchAndCheck( oroModule module )
{
	oroFunction kernel = nullptr;
	CHECK( oroModuleGetFunction( &kernel, module, "userKernel" ) );

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
	return ok ? Result::Passed : Result::WrongValues;
}

// Case 0: a bundle with one SPIR-V module, loaded directly (what the SPIRV example does).
static Result runLoadCase( const char* title, const std::string& path )
{
	std::printf( "%s\n", title );
	std::vector<char> bundle;
	if( !readFile( path, bundle ) ) return Result::Error;
	oroModule module = nullptr;
	CHECK( oroModuleLoadData( &module, bundle.data() ) );
	return launchAndCheck( module );
}

// Cases 1 and 2: SPIR-V modules linked at run time with hipLink*, then loaded.
static Result runLinkCase( const char* title, const std::vector<std::string>& paths )
{
	std::printf( "%s\n", title );
	std::vector<std::vector<char>> modules( paths.size() );
	for( size_t i = 0; i < paths.size(); ++i )
		if( !readFile( paths[i], modules[i] ) ) return Result::Error;

	hipLinkState_t link = nullptr;
	CHECK_HIP( linkCreate( 0, nullptr, nullptr, &link ) );
	for( size_t i = 0; i < modules.size(); ++i )
		CHECK_HIP( linkAddData( link, hipJitInputSpirv, modules[i].data(), modules[i].size(), paths[i].c_str(), 0, nullptr, nullptr ) );
	void*  binary	  = nullptr;
	size_t binarySize = 0;
	CHECK_HIP( linkComplete( link, &binary, &binarySize ) );

	oroModule module = nullptr;
	CHECK( oroModuleLoadData( &module, binary ) );
	CHECK_HIP( linkDestroy( link ) );
	return launchAndCheck( module );
}

int main( int argc, char** argv )
{
	// Diagnostics, set before the HIP runtime is loaded so that comgr picks them up: its log goes to a file whose
	// relevant lines are printed when an API call fails.
	const std::string logPath = executableDir() + "SPIRVLink_comgr.log";
	if( !std::getenv( "AMD_COMGR_REDIRECT_LOGS" ) )
	{
		setEnvironmentVariable( "AMD_COMGR_EMIT_VERBOSE_LOGS", "1" );
		setEnvironmentVariable( "AMD_COMGR_REDIRECT_LOGS", logPath.c_str() );
	}

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

	std::printf( "HIP runtime: %s\n", hipPath.c_str() );

	auto report = [&]( size_t logOffset, Result result ) {
		if( result == Result::Error ) printComgrLog( logPath, logOffset );
		return result == Result::Passed;
	};
	auto run = [&]( const char* title, const std::vector<std::string>& paths ) {
		return report( fileSize( logPath ), runLinkCase( title, paths ) );
	};

	bool ok = true;
	if( argc > 1 )
	{
		ok = run( "Custom case: modules from the command line linked at run time", std::vector<std::string>( argv + 1, argv + argc ) );
	}
	else
	{
		const std::string dir = executableDir() + "SPIRVLink/";
		if( !std::ifstream( dir + "lib.spv", std::ios::binary ).good() )
		{
			std::fprintf( stderr, "Cannot find %slib.spv; the pre-build step compiles it with the HIP SDK compiler\n", dir.c_str() );
			return OROCHI_TEST_RETCODE__ERROR;
		}
		ok = report( fileSize( logPath ), runLoadCase( "Case 0: all.bundle loaded with hipModuleLoadData (SPIR-V JIT, no linking)", dir + "all.bundle" ) ) && ok;
		ok = run( "Case 1: lib.spv + user.spv linked at run time with hipLink*", { dir + "lib.spv", dir + "user.spv" } ) && ok;
		ok = run( "Case 2 (control for case 1): all.spv, both units compiled as one module, via hipLink*", { dir + "all.spv" } ) && ok;
	}

	check( oroCtxDestroy( context ), "oroCtxDestroy" );
	if( ok ) std::remove( logPath.c_str() );
	std::printf( "SPIRVLink: %s\n", ok ? "PASS" : "FAIL" );
	return ok ? OROCHI_TEST_RETCODE__SUCCESS : OROCHI_TEST_RETCODE__ERROR;
}
