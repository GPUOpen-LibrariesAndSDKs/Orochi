#include <Orochi/Orochi.h>
#include <UnitTest/demoErrorCodes.h>

#include <cstdio>
#include <cstdlib>
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
	if( argc != 2 )
	{
		std::fprintf( stderr, "Usage: %s <kernel.bundle>\n", argv[0] );
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

	// Module loading only needs the HIP driver API, not hipRTC.
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

	oroModule module = nullptr;
	oroFunction kernel = nullptr;
	// HIP JIT-compiles the bundled SPIR-V for the selected device.
	CHECK_ORO( oroModuleLoadData( &module, bundle.data() ) );
	CHECK_ORO( oroModuleGetFunction( &kernel, module, "writeValue" ) );

	int value = 0;
	oroDeviceptr deviceValue = nullptr;
	CHECK_ORO( oroMalloc( &deviceValue, sizeof( value ) ) );
	const void* args[] = { &deviceValue };
	CHECK_ORO( oroModuleLaunchKernel( kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr ) );
	CHECK_ORO( oroDeviceSynchronize() );
	CHECK_ORO( oroMemcpyDtoH( &value, deviceValue, sizeof( value ) ) );
	CHECK_ORO( oroFree( deviceValue ) );
	CHECK_ORO( oroModuleUnload( module ) );
	CHECK_ORO( oroCtxDestroy( context ) );

	if( value != 42 )
	{
		std::fprintf( stderr, "FAIL: expected 42, got %d\n", value );
		return OROCHI_TEST_RETCODE__ERROR;
	}
	std::printf( "PASS: value = %d\n", value );
	return OROCHI_TEST_RETCODE__SUCCESS;
}
