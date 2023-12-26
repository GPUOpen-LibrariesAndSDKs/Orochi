#include <ParallelPrimitives/RadixSort.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>

#if defined( ORO_PP_LOAD_FROM_STRING )

// Note: the include order must be in this particular form.
// clang-format off
#include <ParallelPrimitives/cache/Kernels.h>
#include <ParallelPrimitives/cache/KernelArgs.h>
// clang-format on
#endif

#if defined( __GNUC__ )
#include <dlfcn.h>
#endif

constexpr uint64_t div_round_up64( uint64_t val, uint64_t divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }
constexpr uint64_t next_multiple64( uint64_t val, uint64_t divisor ) noexcept { return div_round_up64( val, divisor ) * divisor; }

namespace
{
#if defined( ORO_PRECOMPILED )
constexpr auto useBitCode = true;
#else
constexpr auto useBitCode = false;
#endif

#if defined( ORO_PP_LOAD_FROM_STRING )
constexpr auto useBakeKernel = true;
#else
constexpr auto useBakeKernel = false;
static const char* hip_RadixSortKernels = nullptr;
namespace hip
{
static const char** RadixSortKernelsArgs = nullptr;
static const char** RadixSortKernelsIncludes = nullptr;
} // namespace hip
#endif

static_assert( !( useBitCode && useBakeKernel ), "useBitCode and useBakeKernel cannot coexist" );

#if !defined( __GNUC__ )
const HMODULE GetCurrentModule()
{
	HMODULE hModule = NULL;
	// hModule is NULL if GetModuleHandleEx fails.
	GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCTSTR)GetCurrentModule, &hModule );
	return hModule;
}
#else
void GetCurrentModule1() {}
#endif

void printKernelInfo( const std::string& name, oroFunction func )
{
	std::cout << "Function: " << name;

	int numReg{};
	int sharedSizeBytes{};
	int constSizeBytes{};
	oroFuncGetAttribute( &numReg, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
	oroFuncGetAttribute( &sharedSizeBytes, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
	oroFuncGetAttribute( &constSizeBytes, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
	std::cout << ", vgpr : shared = " << numReg << " : " << sharedSizeBytes << " : " << constSizeBytes << '\n';
}

} // namespace

namespace Oro
{

RadixSort::RadixSort( oroDevice device, OrochiUtils& oroutils, oroStream stream, const std::string& kernelPath, const std::string& includeDir ) : m_device{ device }, m_oroutils{ oroutils }
{
	oroGetDeviceProperties( &m_props, device );
	configure( kernelPath, includeDir, stream );
}

//void RadixSort::exclusiveScanCpu( const Oro::GpuMemory<int>& countsGpu, Oro::GpuMemory<int>& offsetsGpu ) const noexcept
//{
//	const auto buffer_size = countsGpu.size();
//
//	std::vector<int> counts = countsGpu.getData();
//	std::vector<int> offsets( buffer_size );
//
//	int sum = 0;
//	for( int i = 0; i < counts.size(); ++i )
//	{
//		offsets[i] = sum;
//		sum += counts[i];
//	}
//
//	offsetsGpu.copyFromHost( offsets.data(), std::size( offsets ) );
//}

void RadixSort::compileKernels( const std::string& kernelPath, const std::string& includeDir ) noexcept
{
	static constexpr auto defaultKernelPath{ "../ParallelPrimitives/RadixSortKernels.h" };
	static constexpr auto defaultIncludeDir{ "../" };

	const auto currentKernelPath{ ( kernelPath == "" ) ? defaultKernelPath : kernelPath };
	const auto currentIncludeDir{ ( includeDir == "" ) ? defaultIncludeDir : includeDir };

	const auto getCurrentDir = []() noexcept
	{
#if !defined( __GNUC__ )
		HMODULE hm = GetCurrentModule();
		char buff[MAX_PATH];
		GetModuleFileName( hm, buff, MAX_PATH );
#else
		Dl_info info;
		dladdr( (const void*)GetCurrentModule1, &info );
		const char* buff = info.dli_fname;
#endif
		std::string::size_type position = std::string( buff ).find_last_of( "\\/" );
		return std::string( buff ).substr( 0, position ) + "/";
	};

	std::string binaryPath{};
	std::string log{};
	if constexpr( useBitCode )
	{
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		binaryPath = getCurrentDir();
		binaryPath += isAmd ? "oro_compiled_kernels.hipfb" : "oro_compiled_kernels.fatbin";
		log = "loading pre-compiled kernels at path : " + binaryPath;

		//m_num_threads_per_block_for_count = DEFAULT_COUNT_BLOCK_SIZE;
		//m_num_threads_per_block_for_scan = DEFAULT_SCAN_BLOCK_SIZE;
		//m_num_threads_per_block_for_sort = DEFAULT_SORT_BLOCK_SIZE;

		//const auto warp_size = DEFAULT_WARP_SIZE;

		//m_num_warps_per_block_for_sort = m_num_threads_per_block_for_sort / warp_size;
	}
	else
	{
		log = "compiling kernels at path : " + currentKernelPath + " in : " + currentIncludeDir;

		//m_num_threads_per_block_for_count = m_props.maxThreadsPerBlock > 0 ? m_props.maxThreadsPerBlock : DEFAULT_COUNT_BLOCK_SIZE;
		//m_num_threads_per_block_for_scan = m_props.maxThreadsPerBlock > 0 ? m_props.maxThreadsPerBlock : DEFAULT_SCAN_BLOCK_SIZE;
		//m_num_threads_per_block_for_sort = m_props.maxThreadsPerBlock > 0 ? m_props.maxThreadsPerBlock : DEFAULT_SORT_BLOCK_SIZE;

		//const auto warp_size = ( m_props.warpSize != 0 ) ? m_props.warpSize : DEFAULT_WARP_SIZE;

		//m_num_warps_per_block_for_sort = m_num_threads_per_block_for_sort / warp_size;

		//assert( m_num_threads_per_block_for_count % warp_size == 0 );
		//assert( m_num_threads_per_block_for_scan % warp_size == 0 );
		//assert( m_num_threads_per_block_for_sort % warp_size == 0 );
	}

	//m_num_warps_per_block_for_sort = m_num_threads_per_block_for_sort / m_warp_size;

	if( m_flags == Flag::LOG )
	{
		std::cout << log << std::endl;
	}

	const auto includeArg{ "-I" + currentIncludeDir };
	//const auto overwrite_flag = "-DOVERWRITE";
	//const auto count_block_size_param = "-DCOUNT_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_count );
	//const auto scan_block_size_param = "-DSCAN_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_scan );
	//const auto sort_block_size_param = "-DSORT_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_sort );
	//const auto sort_num_warps_param = "-DSORT_NUM_WARPS_PER_BLOCK_VAL=" + std::to_string( m_num_warps_per_block_for_sort );

	std::vector<const char*> opts;
	opts.push_back( includeArg.c_str() );
	//opts.push_back( overwrite_flag );
	//opts.push_back( count_block_size_param.c_str() );
	//opts.push_back( scan_block_size_param.c_str() );
	//opts.push_back( sort_block_size_param.c_str() );
	//opts.push_back( sort_num_warps_param.c_str() );

	struct Record
	{
		std::string kernelName;
		Kernel kernelType;
	};

	//const std::vector<Record> records{
	//	{ "CountKernel", Kernel::COUNT },	 { "ParallelExclusiveScanSingleWG", Kernel::SCAN_SINGLE_WG }, { "ParallelExclusiveScanAllWG", Kernel::SCAN_PARALLEL },	 { "SortKernel", Kernel::SORT },
	//	{ "SortKVKernel", Kernel::SORT_KV }, { "SortSinglePassKernel", Kernel::SORT_SINGLE_PASS },		  { "SortSinglePassKVKernel", Kernel::SORT_SINGLE_PASS_KV },
	//};
	const std::vector<Record> records{
		{ "SortSinglePassKernel", Kernel::SORT_SINGLE_PASS }, { "SortSinglePassKVKernel", Kernel::SORT_SINGLE_PASS_KV },
	};


	for( const auto& record : records )
	{
#if defined( ORO_PP_LOAD_FROM_STRING )
		oroFunctions[record.kernelType] = oroutils.getFunctionFromString( device, hip_RadixSortKernels, currentKernelPath.c_str(), record.kernelName.c_str(), &opts, 1, hip::RadixSortKernelsArgs, hip::RadixSortKernelsIncludes );
#else

		if constexpr( useBitCode )
		{
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromPrecompiledBinary( binaryPath.c_str(), record.kernelName.c_str() );
		}
		else
		{
			const auto includeArg{ "-I" + currentIncludeDir };
			std::vector<const char*> opts;
			opts.push_back( includeArg.c_str() );
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromFile( m_device, currentKernelPath.c_str(), record.kernelName.c_str(), &opts );
		}

#endif
		if( m_flags == Flag::LOG )
		{
			printKernelInfo( record.kernelName, oroFunctions[record.kernelType] );
		}
	}

	// TODO: bit code support?
#define LOAD_FUNC( var, kernel ) var = m_oroutils.getFunctionFromFile( m_device, currentKernelPath.c_str(), kernel, &opts );
	LOAD_FUNC( m_gHistogram, "gHistogram" );
	LOAD_FUNC( m_gPrefixSum, "gPrefixSum" );
	LOAD_FUNC( m_onesweep_reorderKey64, "onesweep_reorderKey64" );
	LOAD_FUNC( m_onesweep_reorderKeyPair64, "onesweep_reorderKeyPair64" );
#undef LOAD_FUNC
	// const auto includeArg{ "-I" + currentIncludeDir };
	// const auto overwrite_flag = "-DOVERWRITE";
	// const auto count_block_size_param = "-DCOUNT_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_count );
	// const auto scan_block_size_param = "-DSCAN_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_scan );
	// const auto sort_block_size_param = "-DSORT_WG_SIZE_VAL=" + std::to_string( m_num_threads_per_block_for_sort );
	// const auto sort_num_warps_param = "-DSORT_NUM_WARPS_PER_BLOCK_VAL=" + std::to_string( m_num_warps_per_block_for_sort );

	// std::vector<const char*> opts;

	// if( const std::string device_name = m_props.name; device_name.find( "NVIDIA" ) != std::string::npos )
	// {
	// 	opts.push_back( "--use_fast_math" );
	// }
	// else
	// {
	// 	opts.push_back( "-ffast-math" );
	// }

	// opts.push_back( includeArg.c_str() );
	// opts.push_back( overwrite_flag );
	// opts.push_back( count_block_size_param.c_str() );
	// opts.push_back( scan_block_size_param.c_str() );
	// opts.push_back( sort_block_size_param.c_str() );
	// opts.push_back( sort_num_warps_param.c_str() );

	// for( const auto& record : records )
	// {
	// 	if constexpr( useBakeKernel )
	// 	{
	// 		oroFunctions[record.kernelType] = m_oroutils.getFunctionFromString( m_device, hip_RadixSortKernels, currentKernelPath.c_str(), record.kernelName.c_str(), &opts, 1, hip::RadixSortKernelsArgs, hip::RadixSortKernelsIncludes );
	// 	}
	// 	else if constexpr( useBitCode )
	// 	{
	// 		oroFunctions[record.kernelType] = m_oroutils.getFunctionFromPrecompiledBinary( binaryPath.c_str(), record.kernelName.c_str() );
	// 	}
	// 	else
	// 	{
	// 		oroFunctions[record.kernelType] = m_oroutils.getFunctionFromFile( m_device, currentKernelPath.c_str(), record.kernelName.c_str(), &opts );
	// 	}

	// 	if( m_flags == Flag::LOG )
	// 	{
	// 		printKernelInfo( record.kernelName, oroFunctions[record.kernelType] );
	// 	}
	// }
}

//int RadixSort::calculateWGsToExecute( const int blockSize ) const noexcept
//{
//	const int warpSize = ( m_props.warpSize != 0 ) ? m_props.warpSize : DEFAULT_WARP_SIZE;
//	const int warpPerWG = blockSize / warpSize;
//	const int warpPerWGP = m_props.maxThreadsPerMultiProcessor / warpSize;
//	const int occupancyFromWarp = ( warpPerWGP > 0 ) ? ( warpPerWGP / warpPerWG ) : 1;
//
//	const int occupancy = std::max( 1, occupancyFromWarp );
//
//	if( m_flags == Flag::LOG )
//	{
//		std::cout << "Occupancy: " << occupancy << '\n';
//	}
//
//	static constexpr auto min_num_blocks = 16;
//	auto number_of_blocks = m_props.multiProcessorCount > 0 ? m_props.multiProcessorCount * occupancy : min_num_blocks;
//
//	if( m_num_threads_per_block_for_scan > BIN_SIZE )
//	{
//		// Note: both are divisible by 2
//		const auto base = m_num_threads_per_block_for_scan / BIN_SIZE;
//
//		// Floor
//		number_of_blocks = ( number_of_blocks / base ) * base;
//	}
//
//	return number_of_blocks;
//}

void RadixSort::configure( const std::string& kernelPath, const std::string& includeDir, oroStream stream ) noexcept
{
	compileKernels( kernelPath, includeDir );

	u64 gpSumBuffer = sizeof( u32 ) * 256 * sizeof( u32 /* key type */ );
	m_gpSumBuffer.resizeAsync( gpSumBuffer, false /*copy*/, stream );

	u64 lookBackBuffer = sizeof( u64 ) * ( 256 * LOOKBACK_TABLE_SIZE );
	m_lookbackBuffer.resizeAsync( lookBackBuffer, false /*copy*/, stream );

	m_tailIterator.resizeAsync( 1, false /*copy*/, stream );
	m_tailIterator.resetAsync( stream );
	m_gpSumCounter.resizeAsync( 1, false /*copy*/, stream );
	//m_num_blocks_for_count = calculateWGsToExecute( m_num_threads_per_block_for_count );

	///// The tmp buffer size of the count kernel and the scan kernel.

	//const auto tmp_buffer_size = BIN_SIZE * m_num_blocks_for_count;

	///// @c tmp_buffer_size must be divisible by @c m_num_threads_per_block_for_scan
	///// This is guaranteed since @c m_num_blocks_for_count will be adjusted accordingly

	//m_num_blocks_for_scan = tmp_buffer_size / m_num_threads_per_block_for_scan;

	//m_tmp_buffer.resizeAsync( tmp_buffer_size, false, stream );

	//if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	//{
	//	// These are for the scan kernel
	//	m_partial_sum.resizeAsync( m_num_blocks_for_scan, false, stream );
	//	m_is_ready.resizeAsync( m_num_blocks_for_scan, false, stream );
	//  m_is_ready.resetAsync( stream );
	//}
}
void RadixSort::setFlag( Flag flag ) noexcept { m_flags = flag; }

void RadixSort::sort( const KeyValueSoA& src, const KeyValueSoA& dst, uint32_t n, int startBit, int endBit, oroStream stream ) noexcept
{
	bool keyPair = src.value != nullptr;

	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		if( keyPair )
		{
			const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS_KV];
			const void* args[] = { &src.key, &src.value, &dst.key, &dst.value, &n, &startBit, &endBit };
			OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		}
		else
		{
			const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS];
			const void* args[] = { &src, &dst, &n, &startBit, &endBit };
			OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		}
		return;
	}

	int nIteration = div_round_up64( endBit - startBit, 8 );
	uint64_t numberOfBlocks = div_round_up64( n, RADIX_SORT_BLOCK_SIZE );

	// Buffers
	void* gpSumBuffer = m_gpSumBuffer.ptr();
	void* lookBackBuffer = m_lookbackBuffer.ptr();
	void* tailIteratorBuffer = m_tailIterator.ptr();

	m_lookbackBuffer.resetAsync( stream );
	m_gpSumCounter.resetAsync( stream );
	m_gpSumBuffer.resetAsync( stream );

	// counter for gHistogram. 
	{
		void* counter = m_gpSumCounter.ptr();
		int maxBlocksPerMP = 0;
		oroError e = oroOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerMP, m_gHistogram, GHISTOGRAM_THREADS_PER_BLOCK, 0 );
		const int nBlocks = e == oroSuccess ? maxBlocksPerMP * m_props.multiProcessorCount : 2048;

		const void* args[] = { &src.key, &n, &gpSumBuffer, &startBit, &counter };
		OrochiUtils::launch1D( m_gHistogram, nBlocks * GHISTOGRAM_THREADS_PER_BLOCK, args, GHISTOGRAM_THREADS_PER_BLOCK, 0, stream );
	}
	{
		const void* args[] = { &gpSumBuffer };
		OrochiUtils::launch1D( m_gPrefixSum, nIteration * 256, args, 256, 0, stream );
	}

	auto s = src;
	auto d = dst;
	for( int i = 0; i < nIteration; i++ )
	{
		if( numberOfBlocks < LOOKBACK_TABLE_SIZE * 2 )
		{
			m_lookbackBuffer.resetAsync( stream );
		} // other wise, we can skip zero clear look back buffer

		if( keyPair )
		{
			const void* args[] = { &s.key, &d.key, &s.value, &d.value, &n, &gpSumBuffer, &lookBackBuffer, &tailIteratorBuffer, & startBit, &i };
			OrochiUtils::launch1D( m_onesweep_reorderKeyPair64, numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		else
		{
			const void* args[] = { &s.key, &d.key, &n, &gpSumBuffer, &lookBackBuffer, &tailIteratorBuffer, &startBit, &i };
			OrochiUtils::launch1D( m_onesweep_reorderKey64, numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		std::swap( s, d );
	}

	if( s.key == src.key )
	{
		oroMemcpyDtoDAsync( (oroDeviceptr)dst.key, (oroDeviceptr)src.key, sizeof( uint32_t ) * n, stream );

		if( keyPair )
		{
			oroMemcpyDtoDAsync( (oroDeviceptr)dst.value, (oroDeviceptr)src.value, sizeof( uint32_t ) * n, stream );
		}
	}
}

void RadixSort::sort( u32* src, u32* dst, uint32_t n, int startBit, int endBit, oroStream stream ) noexcept
{
	sort( KeyValueSoA{ src, nullptr }, KeyValueSoA{ dst, nullptr }, n, startBit, endBit, stream );
}
}; // namespace Oro
