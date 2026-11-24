#include <Orochi/Orochi.h>
#include <UnitTest/demoErrorCodes.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void checkOro( oroError status, const char* call )
{
	if( status == oroSuccess ) return;
	const char* message = nullptr;
	oroGetErrorString( status, &message );
	std::fprintf( stderr, "%s failed (%d): %s\n", call, static_cast<int>( status ), message ? message : "unknown error" );
	std::exit( OROCHI_TEST_RETCODE__ERROR );
}

#define CHECK_ORO( call ) checkOro( call, #call )

int main( int argc, char** argv )
{
	if( argc < 2 || argc > 3 )
	{
		std::fprintf( stderr, "Usage: %s <kernel.bundle> [hip-runtime-library]\n", argv[0] );
		return OROCHI_TEST_RETCODE__ERROR;
	}

	std::ifstream input( argv[1], std::ios::binary | std::ios::ate );
	if( !input || input.tellg() <= 0 )
	{
		std::fprintf( stderr, "Cannot read a nonempty bundle from %s\n", argv[1] );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	const auto size = input.tellg();
	std::vector<char> bundle( static_cast<size_t>( size ) );
	input.seekg( 0 );
	if( !input.read( bundle.data(), static_cast<std::streamsize>( size ) ) )
	{
		std::fprintf( stderr, "Cannot read the complete bundle from %s\n", argv[1] );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	constexpr char bundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
	if( bundle.size() < sizeof( bundleMagic ) - 1 || std::memcmp( bundle.data(), bundleMagic, sizeof( bundleMagic ) - 1 ) != 0 )
	{
		std::fprintf( stderr, "Expected an uncompressed offload bundle; compile without --no-gpu-bundle-output\n" );
		return OROCHI_TEST_RETCODE__ERROR;
	}

	// Module loading only needs the HIP driver API, not hipRTC.
	const char* hipPaths[] = { argc == 3 ? argv[2] : nullptr, nullptr };
	const int initStatus = oroInitialize( ORO_API_HIPDRIVER, 0, argc == 3 ? hipPaths : nullptr );
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

	oroDeviceProp properties{};
	CHECK_ORO( oroGetDeviceProperties( &properties, device ) );
	int runtimeVersion = 0;
	CHECK_ORO( oroRuntimeGetVersion( &runtimeVersion ) );
	std::printf( "HIP backend: %s (%s), runtime %d\n", properties.name, properties.gcnArchName, runtimeVersion );
	std::fflush( stdout );

	oroModule module = nullptr;
	oroFunction kernel = nullptr;
	// HIP JIT-compiles the bundled SPIR-V for the selected device.
	CHECK_ORO( oroModuleLoadData( &module, bundle.data() ) );
	CHECK_ORO( oroModuleGetFunction( &kernel, module, "transform" ) );

	const unsigned count = 257;
	std::array<int, count> values;
	for( unsigned i = 0; i < count; ++i )
		values[i] = static_cast<int>( i ) - 128;
	const size_t bytes = values.size() * sizeof( int );
	oroDeviceptr deviceValues = nullptr;
	CHECK_ORO( oroMalloc( &deviceValues, bytes ) );
	CHECK_ORO( oroMemcpyHtoD( deviceValues, values.data(), bytes ) );
	const void* args[] = { &deviceValues, &count };
	CHECK_ORO( oroModuleLaunchKernel( kernel, ( count + 63 ) / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr ) );
	CHECK_ORO( oroDeviceSynchronize() );
	CHECK_ORO( oroMemcpyDtoH( values.data(), deviceValues, bytes ) );
	CHECK_ORO( oroFree( deviceValues ) );
	CHECK_ORO( oroModuleUnload( module ) );
	CHECK_ORO( oroCtxDestroy( context ) );

	for( unsigned i = 0; i < count; ++i )
	{
		const int expected = 3 * ( static_cast<int>( i ) - 128 ) + 7;
		if( values[i] != expected )
		{
			std::fprintf( stderr, "FAIL at index %u: expected %d, got %d\n", i, expected, values[i] );
			return OROCHI_TEST_RETCODE__ERROR;
		}
	}
	std::printf( "PASS: all %u results match\n", count );
	return OROCHI_TEST_RETCODE__SUCCESS;
}
