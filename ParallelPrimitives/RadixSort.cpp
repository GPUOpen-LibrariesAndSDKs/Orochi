#include <ParallelPrimitives/RadixSort.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

#if defined( ORO_PP_LOAD_FROM_STRING )

// Note: the include order must be in this particular form.
// clang-format off
#include <ParallelPrimitives/cache/Kernels.h>
#include <ParallelPrimitives/cache/KernelArgs.h>
// clang-format on
#endif

namespace
{
#if defined( ORO_PRECOMPILED )
constexpr auto useBitCode = true;
#else
constexpr auto useBitCode = false;
#endif

void printKernelInfo( oroFunction func )
{
	int numReg{};
	int sharedSizeBytes{};
	int constSizeBytes{};
	oroFuncGetAttribute( &numReg, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
	oroFuncGetAttribute( &sharedSizeBytes, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
	oroFuncGetAttribute( &constSizeBytes, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
	std::cout << "vgpr : shared = " << numReg << " : "
			  << " : " << sharedSizeBytes << " : " << constSizeBytes << '\n';
}

} // namespace

namespace Oro
{

RadixSort::RadixSort()
{
	if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		m_partialSum.resize( m_nWGsToExecute );
		OrochiUtils::malloc( m_isReady, m_nWGsToExecute );
		OrochiUtils::memset( m_isReady, false, m_nWGsToExecute * sizeof( bool ) );
	}
}

RadixSort::~RadixSort()
{
	if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_isReady );
	}
}

void RadixSort::exclusiveScanCpu( int* countsGpu, int* offsetsGpu, const int nWGsToExecute, oroStream stream ) noexcept
{
	std::vector<int> counts( Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::copyDtoHAsync( counts.data(), countsGpu, Oro::BIN_SIZE * nWGsToExecute, stream );
	OrochiUtils::waitForCompletion( stream );

	std::vector<int> offsets( Oro::BIN_SIZE * nWGsToExecute );

	int sum = 0;
	for( int i = 0; i < counts.size(); ++i )
	{
		offsets[i] = sum;
		sum += counts[i];
	}

	OrochiUtils::copyHtoDAsync( offsetsGpu, offsets.data(), Oro::BIN_SIZE * nWGsToExecute, stream );
	OrochiUtils::waitForCompletion( stream );
}

void RadixSort::compileKernels( oroDevice device, OrochiUtils& oroutils, const std::string& kernelPath, const std::string& includeDir ) noexcept
{
	constexpr auto defaultKernelPath{ "../ParallelPrimitives/RadixSortKernels.h" };
	constexpr auto defaultIncludeDir{ "../" };

	const auto currentKernelPath{ ( kernelPath == "" ) ? defaultKernelPath : kernelPath };
	const auto currentIncludeDir{ ( includeDir == "" ) ? defaultIncludeDir : includeDir };

	std::string binaryPath{};
	if constexpr( useBitCode )
	{
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		binaryPath = isAmd ? "../bitcodes/oro_compiled_kernels.hipfb" : "../bitcodes/oro_compiled_kernels.fatbin";
		if( m_flags == Flag::LOG )
		{
			std::cout << "loading pre-compiled kernels at path : " << binaryPath << '\n';
		}
	}
	else
	{
		if( m_flags == Flag::LOG )
		{
			std::cout << "compiling kernels at path : " << currentKernelPath << " in : " << currentIncludeDir << '\n';
		}
	}

	const auto includeArg{ "-I" + currentIncludeDir };

	std::vector<const char*> opts;
	opts.push_back( includeArg.c_str() );

	struct Record
	{
		std::string kernelName;
		Kernel kernelType;
	};

	const std::vector<Record> records{
		{ "CountKernel", Kernel::COUNT }, { "CountKernelReference", Kernel::COUNT_REF }, { "ParallelExclusiveScanSingleWG", Kernel::SCAN_SINGLE_WG }, { "ParallelExclusiveScanAllWG", Kernel::SCAN_PARALLEL },
		{ "SortKernel", Kernel::SORT },	  { "SortKVKernel", Kernel::SORT_KV },			 { "SortSinglePassKernel", Kernel::SORT_SINGLE_PASS },		  { "SortSinglePassKVKernel", Kernel::SORT_SINGLE_PASS_KV },
	};

	for( const auto& record : records )
	{
#if defined( ORO_PP_LOAD_FROM_STRING )
		oroFunctions[record.kernelType] = oroutils.getFunctionFromString( device, hip_RadixSortKernels, currentKernelPath.c_str(), record.kernelName.c_str(), &opts, 1, hip::RadixSortKernelsArgs, hip::RadixSortKernelsIncludes );
#else

		if constexpr( useBitCode )
		{
			oroFunctions[record.kernelType] = oroutils.getFunctionFromPrecompiledBinary( binaryPath.c_str(), record.kernelName.c_str() );
		}
		else
		{

			oroFunctions[record.kernelType] = oroutils.getFunctionFromFile( device, currentKernelPath.c_str(), record.kernelName.c_str(), &opts );
		}

#endif
		if( m_flags == Flag::LOG )
		{
			printKernelInfo( oroFunctions[record.kernelType] );
		}
	}
}

int RadixSort::calculateWGsToExecute( oroDevice device ) noexcept
{
	oroDeviceProp props{};
	oroGetDeviceProperties( &props, device );

	constexpr auto maxWGSize = std::max( { COUNT_WG_SIZE, SCAN_WG_SIZE, SORT_WG_SIZE } );
	const int warpSize = ( props.warpSize != 0 ) ? props.warpSize : 32;
	const int warpPerWG = maxWGSize / warpSize;
	const int warpPerWGP = props.maxThreadsPerMultiProcessor / warpSize;
	const int occupancyFromWarp = ( warpPerWGP > 0 ) ? ( warpPerWGP / warpPerWG ) : 1;

	// From the runtime measurements this yields better result.
	const int occupancy = std::max( 1, occupancyFromWarp / 2 );

	if( m_flags == Flag::LOG ) std::cout << "Occupancy: " << occupancy << '\n';

	return props.multiProcessorCount * occupancy;
}

RadixSort::u32 RadixSort::configure( oroDevice device, OrochiUtils& oroutils, const std::string& kernelPath, const std::string& includeDir, oroStream stream ) noexcept
{
	compileKernels( device, oroutils, kernelPath, includeDir );
	const auto newWGsToExecute{ calculateWGsToExecute( device ) };

	if( newWGsToExecute != m_nWGsToExecute && selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		m_partialSum.resize( newWGsToExecute );
		OrochiUtils::free( m_isReady );
		OrochiUtils::malloc( m_isReady, newWGsToExecute );
		OrochiUtils::memsetAsync( m_isReady, false, newWGsToExecute * sizeof( bool ), stream );
	}

	m_nWGsToExecute = newWGsToExecute;
	return static_cast<u32>( BIN_SIZE * m_nWGsToExecute );
}
void RadixSort::setFlag( Flag flag ) noexcept { m_flags = flag; }

void RadixSort::sort( const KeyValueSoA src, const KeyValueSoA dst, int n, int startBit, int endBit, u32* tempBuffer, oroStream stream ) noexcept
{
	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS_KV];
		const void* args[] = { &src.key, &src.value, &dst.key, &dst.value, &n, &startBit, &endBit };
		OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		return;
	}

	auto* s{ &src };
	auto* d{ &dst };

	for( int i = startBit; i < endBit; i += N_RADIX )
	{
		sort1pass( *s, *d, n, i, i + std::min( N_RADIX, endBit - i ), (int*)tempBuffer, stream );

		std::swap( s, d );
	}

	if( s == &src )
	{
		OrochiUtils::copyDtoDAsync( dst.key, src.key, n, stream );
		OrochiUtils::copyDtoDAsync( dst.value, src.value, n, stream );
	}
}

void RadixSort::sort( const u32* src, const u32* dst, int n, int startBit, int endBit, u32* tempBuffer, oroStream stream ) noexcept
{
	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS];
		const void* args[] = { &src, &dst, &n, &startBit, &endBit };
		OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		return;
	}

	auto* s{ &src };
	auto* d{ &dst };

	for( int i = startBit; i < endBit; i += N_RADIX )
	{
		sort1pass( *s, *d, n, i, i + std::min( N_RADIX, endBit - i ), (int*)tempBuffer, stream );

		std::swap( s, d );
	}

	if( s == &src )
	{
		OrochiUtils::copyDtoDAsync( dst, src, n, stream );
	}
}

}; // namespace Oro
