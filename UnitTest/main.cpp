#include <gtest/gtest.h>
#include <Orochi/Orochi.h>

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


int main( int argc, char* argv[] ) 
{
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
