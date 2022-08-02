#include <gtest/gtest.h>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>

#define OROASSERT( x ) ASSERT_TRUE( x )
#define OROCHECK( x ) { oroError e = x; OROASSERT( e == ORO_SUCCESS ); }


class OroTestBase : public ::testing::Test
{
  public:
	void SetUp() 
	{
		const int deviceIndex = 0;
		oroApi api = ( oroApi )( ORO_API_CUDA | ORO_API_HIP );
		int a = oroInitialize( api, 0 );
		OROASSERT( a == 0 );

		OROCHECK( oroInit( 0 ) );
		OROCHECK( oroDeviceGet( &m_device, deviceIndex ) );
		OROCHECK( oroCtxCreate( &m_ctx, 0, m_device ) );
	}

	void TearDown() 
	{ 
		OROCHECK( oroCtxDestroy( m_ctx ) );
	}

  protected:
	oroDevice m_device;
	oroCtx m_ctx;

};


TEST_F( OroTestBase, init )
{ 

}

TEST_F( OroTestBase, deviceprops )
{
	{
		oroDeviceProp props;
		OROCHECK( oroGetDeviceProperties( &props, m_device ) );
		printf( "executing on %s (%s)\n", props.name, props.gcnArchName );
		printf( "%d multiProcessors\n", props.multiProcessorCount );
	}
}

TEST_F( OroTestBase, kernelExec ) 
{
	int a_host = -1;
	int* a_device = nullptr;
	OROCHECK( oroMalloc( (oroDeviceptr*)&a_device, sizeof( int ) ) );
	OROCHECK( oroMemset( (oroDeviceptr)a_device, 0, sizeof( int ) ) );
	oroFunction kernel = OrochiUtils::getFunctionFromFile( m_device, "../UnitTest/testKernel.h", "testKernel", 0 ); 
	const void* args[] = { &a_device };
	OrochiUtils::launch1D( kernel, 64, args, 64 );
	OrochiUtils::waitForCompletion();
	OROCHECK( oroMemcpyDtoH( &a_host, (oroDeviceptr)a_device, sizeof( int ) ) );
	OROASSERT( a_host == 2016 );
	OROCHECK( oroFree( (oroDeviceptr)a_device ) );
}

TEST_F( OroTestBase, kernelCompileMultiple )
{
	int a_host = -1;
	int* a_device = nullptr;
	OROCHECK( oroMalloc( (oroDeviceptr*)&a_device, sizeof( int ) ) );
	OROCHECK( oroMemset( (oroDeviceptr)a_device, 0, sizeof( int ) ) );
	std::vector<const char*> additionalSourcePaths = { "../UnitTest/testFunc.cpp" };
	oroFunction kernel = OrochiUtils::getFunctionFromFile( m_device, "../UnitTest/testKernel1.h", "testKernel", 0, &additionalSourcePaths, "../" );
	const void* args[] = { &a_device };
	OrochiUtils::launch1D( kernel, 64, args, 64 );
	OrochiUtils::waitForCompletion();
	OROCHECK( oroMemcpyDtoH( &a_host, (oroDeviceptr)a_device, sizeof( int ) ) );
	OROASSERT( a_host == 2016 );
	OROCHECK( oroFree( (oroDeviceptr)a_device ) );
}

TEST_F( OroTestBase, getErrorString )
{
	oroError error = (oroError)1;
	const char *str = nullptr;
	OROCHECK( oroGetErrorString( error, &str ) );
	oroApi api = oroGetCurAPI( 0 );
	if(api == ORO_API_CUDADRIVER)
		OROASSERT( str != nullptr );
	else if( api == ORO_API_HIP )
		OROASSERT( !strcmp(str, "hipErrorInvalidValue") );
}


int main( int argc, char* argv[] ) 
{
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
