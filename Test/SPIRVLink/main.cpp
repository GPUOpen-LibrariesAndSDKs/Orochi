#include <Orochi/Orochi.h>
#include <UnitTest/demoErrorCodes.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

// Links the given AMDGCN SPIR-V modules at run time with hipLink* and runs `userKernel`, which calls a
// function defined in another module that returns a 12-byte struct by value. Expected output: 2 10 40.
//
// Orochi does not wrap hipLink* yet, so the four entry points are resolved from the HIP runtime that
// hipew already loaded. Linux only.

static void checkOro( oroError status, const char* call )
{
	if( status == oroSuccess ) return;
	const char* message = nullptr;
	oroGetErrorString( status, &message );
	std::fprintf( stderr, "%s failed (%d): %s\n", call, static_cast<int>( status ), message ? message : "unknown error" );
	std::exit( OROCHI_TEST_RETCODE__ERROR );
}

#define CHECK_ORO( call ) checkOro( call, #call )

typedef struct ihipLinkState_t* hipLinkState_t;
constexpr int hipJitInputSpirv = 103;

typedef hipError_t ( *PFN_hipLinkCreate )( unsigned int, void*, void**, hipLinkState_t* );
typedef hipError_t ( *PFN_hipLinkAddData )( hipLinkState_t, int, void*, size_t, const char*, unsigned int, void*, void** );
typedef hipError_t ( *PFN_hipLinkComplete )( hipLinkState_t, void**, size_t* );
typedef hipError_t ( *PFN_hipLinkDestroy )( hipLinkState_t );

template <typename T>
static T resolve( void* library, const char* name )
{
	T fn = reinterpret_cast<T>( dlsym( library, name ) );
	if( !fn )
	{
		std::fprintf( stderr, "Cannot resolve %s from the HIP runtime\n", name );
		std::exit( OROCHI_TEST_RETCODE__ERROR );
	}
	return fn;
}

int main( int argc, char** argv )
{
	if( argc < 2 )
	{
		std::fprintf( stderr, "Usage: %s <module.spv>...\n", argv[0] );
		return OROCHI_TEST_RETCODE__ERROR;
	}

	std::vector<std::vector<char>> modules;
	for( int i = 1; i < argc; ++i )
	{
		std::ifstream input( argv[i], std::ios::binary );
		modules.emplace_back( ( std::istreambuf_iterator<char>( input ) ), std::istreambuf_iterator<char>() );
		if( modules.back().empty() )
		{
			std::fprintf( stderr, "Cannot read %s\n", argv[i] );
			return OROCHI_TEST_RETCODE__ERROR;
		}
	}

	const int initStatus = oroInitialize( ORO_API_HIPDRIVER, 0 );
	if( initStatus != ORO_SUCCESS )
	{
		std::fprintf( stderr, "Cannot load the HIP runtime (%d)\n", initStatus );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	CHECK_ORO( oroInit( 0 ) );
	oroDevice device;
	CHECK_ORO( oroDeviceGet( &device, 0 ) );
	oroCtx context = nullptr;
	CHECK_ORO( oroCtxCreate( &context, 0, device ) );
	CHECK_ORO( oroCtxSetCurrent( context ) );

	// The HIP runtime library is the one that provides hipModuleLoadData to hipew.
	Dl_info info{};
	if( !dladdr( reinterpret_cast<void*>( hipModuleLoadData ), &info ) || !info.dli_fname )
	{
		std::fprintf( stderr, "Cannot locate the HIP runtime library\n" );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	void* hip = dlopen( info.dli_fname, RTLD_NOW | RTLD_NOLOAD );
	if( !hip )
	{
		std::fprintf( stderr, "Cannot reopen %s\n", info.dli_fname );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	auto linkCreate	  = resolve<PFN_hipLinkCreate>( hip, "hipLinkCreate" );
	auto linkAddData  = resolve<PFN_hipLinkAddData>( hip, "hipLinkAddData" );
	auto linkComplete = resolve<PFN_hipLinkComplete>( hip, "hipLinkComplete" );
	auto linkDestroy  = resolve<PFN_hipLinkDestroy>( hip, "hipLinkDestroy" );

	hipLinkState_t link = nullptr;
	CHECK_ORO( static_cast<oroError>( linkCreate( 0, nullptr, nullptr, &link ) ) );
	for( size_t i = 0; i < modules.size(); ++i )
		CHECK_ORO( static_cast<oroError>(
			linkAddData( link, hipJitInputSpirv, modules[i].data(), modules[i].size(), argv[i + 1], 0, nullptr, nullptr ) ) );
	void*  binary	  = nullptr;
	size_t binarySize = 0;
	CHECK_ORO( static_cast<oroError>( linkComplete( link, &binary, &binarySize ) ) );

	oroModule	module = nullptr;
	oroFunction kernel = nullptr;
	CHECK_ORO( oroModuleLoadData( &module, binary ) );
	CHECK_ORO( oroModuleGetFunction( &kernel, module, "userKernel" ) );
	CHECK_ORO( static_cast<oroError>( linkDestroy( link ) ) );

	int			 values[3] = { 0, 0, 0 };
	oroDeviceptr deviceValues = nullptr;
	CHECK_ORO( oroMalloc( &deviceValues, sizeof( values ) ) );
	CHECK_ORO( oroMemsetD8( deviceValues, 0, sizeof( values ) ) );
	const void* args[] = { &deviceValues };
	CHECK_ORO( oroModuleLaunchKernel( kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr ) );
	CHECK_ORO( oroDeviceSynchronize() );
	CHECK_ORO( oroMemcpyDtoH( values, deviceValues, sizeof( values ) ) );
	CHECK_ORO( oroFree( deviceValues ) );
	CHECK_ORO( oroModuleUnload( module ) );
	CHECK_ORO( oroCtxDestroy( context ) );

	const bool ok = values[0] == 2 && values[1] == 10 && values[2] == 40;
	std::printf(
		"%d module(s): out = %d %d %d (expected 2 10 40): %s\n",
		argc - 1,
		values[0],
		values[1],
		values[2],
		ok ? "PASS" : "FAIL" );
	return ok ? OROCHI_TEST_RETCODE__SUCCESS : OROCHI_TEST_RETCODE__ERROR;
}
