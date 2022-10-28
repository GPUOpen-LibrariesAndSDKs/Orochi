#include <gtest/gtest.h>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <fstream>

#if defined( OROASSERT )
	#undef OROASSERT
#endif
#define OROASSERT( x ) ASSERT_TRUE( x )
#define OROCHECK( x ) { oroError e = x; OROASSERT( e == ORO_SUCCESS ); }
#define ORORTCCHECK( x ) { OROASSERT( x == ORORTC_SUCCESS ); }


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
	OrochiUtils o;
	int a_host = -1;
	int* a_device = nullptr;
	OROCHECK( oroMalloc( (oroDeviceptr*)&a_device, sizeof( int ) ) );
	OROCHECK( oroMemset( (oroDeviceptr)a_device, 0, sizeof( int ) ) );
	oroFunction kernel = o.getFunctionFromFile( m_device, "../UnitTest/testKernel.h", "testKernel", 0 ); 
	const void* args[] = { &a_device };
	OrochiUtils::launch1D( kernel, 64, args, 64 );
	OrochiUtils::waitForCompletion();
	OROCHECK( oroMemcpyDtoH( &a_host, (oroDeviceptr)a_device, sizeof( int ) ) );
	OROASSERT( a_host == 2016 );
	OROCHECK( oroFree( (oroDeviceptr)a_device ) );
}

void loadFile( const char* path, std::vector<char>& dst ) 
{
	std::fstream f( path, std::ios::binary | std::ios::in );
	if( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = (size_t)f.tellg();
		dst.resize( size );
		f.seekg( 0, std::fstream::beg );
		f.read( dst.data(), size );
		f.close();
	}
}

TEST_F( OroTestBase, linkBc )
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	int v;
	oroDriverGetVersion( &v );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	std::string archName(props.gcnArchName);
	archName = archName.substr( 0, archName.find( ':' ));
	// todo - generate cubin for NVIDIA GPUs (skip on CUDA for now)
	{
		std::string bcFile = isAmd ? ( "../UnitTest/bitcodes/moduleTestFunc-hip-amdgcn-amd-amdhsa-" + archName + ".bc" ) : "../UnitTest/bitcodes/moduleTestFunc.cubin";
		loadFile( bcFile.c_str(), data1 );
	}
	{
		std::string bcFile = isAmd ? ( "../UnitTest/bitcodes/moduleTestKernel-hip-amdgcn-amd-amdhsa-" + archName + ".bc" ) : "../UnitTest/bitcodes/moduleTestKernel.cubin";
		loadFile( bcFile.c_str(), data0 );
	}

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[6];
		void* option_vals[6];
		float wall_time;

		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size );//todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize = 0;
		orortcJITInputType type = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;
		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		ee = oroModuleGetFunction( &function, module, "testKernel" );
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link ) 
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;

	std::vector<const char*> opts = isAmd ? std::vector<const char *>({ "-fgpu-rdc", "-c", "--cuda-device-only" })
											:  std::vector<const char *>({ "--device-c", "-arch=sm_80" });
	{
		std::string code;
		OrochiUtils::readSourceCode( "../UnitTest/moduleTestKernel.h", code );
		OrochiUtils::getData( m_device, code.c_str(), "../UnitTest/moduleTestKernel.h", &opts, data1 );
	}
	{
		std::string code;
		OrochiUtils::readSourceCode( "../UnitTest/moduleTestFunc.h", code );
		OrochiUtils::getData( m_device, code.c_str(), "../UnitTest/moduleTestFunc.h", &opts, data0 );
	}

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[6];
		void* option_vals[6];
		float wall_time;
		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size );//todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize;

		orortcJITInputType type = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;

		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );
		
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		OrochiUtils::waitForCompletion();
		ee = oroModuleGetFunction( &function, module, "testKernel" );
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link_addFile )
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	std::string archName( props.gcnArchName );
	archName = archName.substr( 0, archName.find( ':' ) );

	std::string bcFile1 = isAmd ? ( "../UnitTest/bitcodes/moduleTestFunc-hip-amdgcn-amd-amdhsa-" + archName + ".bc" )
		: "../UnitTest/bitcodes/moduleTestFunc.cubin";
	std::string bcFile2 = isAmd ? ( "../UnitTest/bitcodes/moduleTestKernel-hip-amdgcn-amd-amdhsa-" + archName + ".bc" )
		: "../UnitTest/bitcodes/moduleTestKernel.cubin";

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[6];
		void* option_vals[6];
		float wall_time;
		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size );//todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize;

		orortcJITInputType type = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;

		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );

		ORORTCCHECK( orortcLinkAddFile( rtc_link_state, type, bcFile1.c_str(), 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddFile( rtc_link_state, type, bcFile2.c_str(), 0, 0, 0 ) );
		// getting HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE here
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		OrochiUtils::waitForCompletion();
		ee = oroModuleGetFunction( &function, module, "testKernel" );
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link_null_name ) 
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;

	std::vector<const char*> opts = isAmd ? std::vector<const char *>({ "-fgpu-rdc", "-c", "--cuda-device-only" })
											:  std::vector<const char *>({ "--device-c", "-arch=sm_80" });
	{
		std::string code;
		OrochiUtils::readSourceCode( "../UnitTest/moduleTestKernel.h", code );
		OrochiUtils::getData( m_device, code.c_str(), "../UnitTest/moduleTestKernel.h", &opts, data1 );
	}
	{
		std::string code;
		OrochiUtils::readSourceCode( "../UnitTest/moduleTestFunc.h", code );
		OrochiUtils::getData( m_device, code.c_str(), "../UnitTest/moduleTestFunc.h", &opts, data0 );
	}

	{
		orortcLinkState rtc_link_state;

		void* binary;
		size_t binarySize;
		orortcJITInputType type = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;

		ORORTCCHECK( orortcLinkCreate( 0, 0, 0, &rtc_link_state ) );
		
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		ee = oroModuleGetFunction( &function, module, "testKernel" );
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link_bundledBc )
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	int v;
	oroDriverGetVersion( &v );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;

	{
		std::string bcFile = isAmd ? "../UnitTest/bitcodes/moduleTestFunc-hip-amdgcn-amd-amdhsa.bc" : "../UnitTest/bitcodes/moduleTestFunc.fatbin";
		loadFile( bcFile.c_str(), data1 );
	}
	{
		std::string bcFile = isAmd ? "../UnitTest/bitcodes/moduleTestKernel-hip-amdgcn-amd-amdhsa.bc" : "../UnitTest/bitcodes/moduleTestKernel.fatbin";
		loadFile( bcFile.c_str(), data0 );
	}

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[7];
		void* option_vals[7];
		float wall_time;

		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size ); // todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize = 0;
		const orortcJITInputType type = isAmd ? ORORTC_JIT_INPUT_LLVM_BUNDLED_BITCODE : ORORTC_JIT_INPUT_FATBINARY;
		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		ORORTCCHECK(oroModuleGetFunction( &function, module, "testKernel" ));
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link_bundledBc_with_bc )
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	int v;
	oroDriverGetVersion( &v );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;

	{
		std::string bcFile = isAmd ? "../UnitTest/bitcodes/moduleTestFunc-hip-amdgcn-amd-amdhsa.bc" : "../UnitTest/bitcodes/moduleTestFunc.fatbin";
		loadFile( bcFile.c_str(), data1 );
	}
	{
		std::vector<const char*> opts = isAmd ? std::vector<const char *>({ "-fgpu-rdc", "-c", "--cuda-device-only" })
											:  std::vector<const char *>({ "--device-c", "-arch=sm_80" });
		std::string code;
		OrochiUtils::readSourceCode( "../UnitTest/moduleTestKernel.h", code );
		OrochiUtils::getData( m_device, code.c_str(), "../UnitTest/moduleTestKernel.h", &opts, data0 );
	}

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[7];
		void* option_vals[7];
		float wall_time;

		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size ); // todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize = 0;
		const orortcJITInputType type0 = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;
		const orortcJITInputType type1 = isAmd ? ORORTC_JIT_INPUT_LLVM_BUNDLED_BITCODE : ORORTC_JIT_INPUT_FATBINARY;
		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type1, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type0, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		ORORTCCHECK(oroModuleGetFunction( &function, module, "testKernel" ));
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, link_bundledBc_with_bc_loweredName )
{
	oroDeviceProp props;
	OROCHECK( oroGetDeviceProperties( &props, m_device ) );
	int v;
	oroDriverGetVersion( &v );
	std::vector<char> data0;
	std::vector<char> data1;
	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	const char* funcName = "testKernel<0>";
	std::string loweredNameStr;
	orortcProgram prog;

	{
		std::string bcFile = isAmd ? "../UnitTest/bitcodes/moduleTestFunc-hip-amdgcn-amd-amdhsa.bc" : "../UnitTest/bitcodes/moduleTestFunc.fatbin";
		loadFile( bcFile.c_str(), data1 );
	}
	{
		std::vector<const char*> opts = isAmd ? std::vector<const char*>( { "-fgpu-rdc", "-c", "--cuda-device-only" } ) : std::vector<const char*>( { "--device-c", "-arch=sm_80" } );
		std::string code;

		OrochiUtils::readSourceCode( "../UnitTest/moduleTestKernel_loweredName.h", code );
		OrochiUtils::getProgram( m_device, code.c_str(), "../UnitTest/moduleTestKernel_loweredName.h", &opts, funcName, &prog );
		const char* loweredName = 0;
		ORORTCCHECK( orortcGetLoweredName( prog, funcName, &loweredName ) );
		loweredNameStr = std::string( loweredName );
		size_t codeSize;
		ORORTCCHECK( orortcGetBitcodeSize( prog, &codeSize ) );

		data0.resize( codeSize );
		ORORTCCHECK( orortcGetBitcode( prog, data0.data() ) );
		ORORTCCHECK( orortcDestroyProgram( &prog ) );
	}

	{
		orortcLinkState rtc_link_state;
		orortcJIT_option options[7];
		void* option_vals[7];
		float wall_time;

		unsigned int log_size = 8192;
		char error_log[8192];
		char info_log[8192];
		size_t out_size;
		void* cuOut;

		options[0] = ORORTC_JIT_WALL_TIME;
		option_vals[0] = (void*)( &wall_time );

		options[1] = ORORTC_JIT_INFO_LOG_BUFFER;
		option_vals[1] = info_log;

		options[2] = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		option_vals[2] = (void*)( log_size );

		options[3] = ORORTC_JIT_ERROR_LOG_BUFFER;
		option_vals[3] = error_log;

		options[4] = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		option_vals[4] = (void*)( log_size ); // todo. behavior difference

		options[5] = ORORTC_JIT_LOG_VERBOSE;
		option_vals[5] = (void*)1;

		void* binary;
		size_t binarySize = 0;
		const orortcJITInputType type0 = isAmd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_CUBIN;
		const orortcJITInputType type1 = isAmd ? ORORTC_JIT_INPUT_LLVM_BUNDLED_BITCODE : ORORTC_JIT_INPUT_FATBINARY;
		ORORTCCHECK( orortcLinkCreate( 6, options, option_vals, &rtc_link_state ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type1, data1.data(), data1.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkAddData( rtc_link_state, type0, data0.data(), data0.size(), 0, 0, 0, 0 ) );
		ORORTCCHECK( orortcLinkComplete( rtc_link_state, &binary, &binarySize ) );

		oroFunction function;
		oroModule module;
		oroError ee = oroModuleLoadData( &module, binary );
		oroError e =  oroModuleGetFunction( &function, module, loweredNameStr.c_str() );
		int x_host = -1;
		int* x_device = nullptr;
		OROCHECK( oroMalloc( (oroDeviceptr*)&x_device, sizeof( int ) ) );
		OROCHECK( oroMemset( (oroDeviceptr)x_device, 0, sizeof( int ) ) );
		const void* args[] = { &x_device };

		OrochiUtils::launch1D( function, 64, args, 64 );
		OrochiUtils::waitForCompletion();
		OROCHECK( oroMemcpyDtoH( &x_host, (oroDeviceptr)x_device, sizeof( int ) ) );
		OROASSERT( x_host == 2016 );
		OROCHECK( oroFree( (oroDeviceptr)x_device ) );
		ORORTCCHECK( orortcLinkDestroy( rtc_link_state ) );
		ORORTCCHECK( oroModuleUnload( module ) );
	}
}

TEST_F( OroTestBase, getErrorString )
{
	oroError error = oroError::oroErrorInvalidValue;
	const char* str = nullptr;
	OROCHECK( oroGetErrorString( error, &str ) );
	const oroApi api = oroGetCurAPI( 0 );
	if( api == ORO_API_CUDADRIVER )
	{
		OROASSERT( str != nullptr );
	}
	else if( api == ORO_API_HIP )
	{
		constexpr auto hipErrorMessage{ "hipErrorInvalidValue" };
		OROASSERT( std::string( str ) == hipErrorMessage );
	}
	else
	{
		// Unsupported api. This should not happen.
		OROASSERT( false );
	}
}

int main( int argc, char* argv[] )
{
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}
