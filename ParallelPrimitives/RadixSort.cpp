#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSort.h>
#include <Test/Stopwatch.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

namespace
{
/// @brief Exclusive scan algorithm on CPU for testing.
/// It copies the count result from the Device to Host before computation, and then copies the offsets back from Host to Device afterward.
/// @param countsGpu The count result in GPU memory. Otuput: The offset.
/// @param offsetsGpu The offsets.
/// @param nWGsToExecute Number of WGs to execute
void exclusiveScanCpu( int* countsGpu, int* offsetsGpu, const int nWGsToExecute ) noexcept
{
	std::vector<int> counts( Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::copyDtoH( counts.data(), countsGpu, Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();

	std::vector<int> offsets( Oro::BIN_SIZE * nWGsToExecute );
	std::exclusive_scan( std::cbegin( counts ), std::cend( counts ), std::begin( offsets ), 0 );

	OrochiUtils::copyHtoD( offsetsGpu, offsets.data(), Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();
}

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
		OrochiUtils::malloc( m_partialSum, m_nWGsToExecute );
		OrochiUtils::malloc( m_isReady, m_nWGsToExecute );
		OrochiUtils::memset( m_isReady, false, m_nWGsToExecute * sizeof( bool ) );
	}
}

RadixSort::~RadixSort()
{
	if( selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_partialSum );
		OrochiUtils::free( m_isReady );
	}
}

void RadixSort::compileKernels( oroDevice device, const std::string& kernelPath, const std::string& includeDir ) noexcept
{
	constexpr auto defaultKernelPath{ "../ParallelPrimitives/RadixSortKernels.h" };
	constexpr auto defaultIncludeDir{ "../" };

	const auto currentKernelPath{ ( kernelPath == "" ) ? defaultKernelPath : kernelPath };
	const auto currentIncludeDir{ ( includeDir == "" ) ? defaultIncludeDir : includeDir };

	if( m_flags == Flag::LOG )
	{
		std::cout << "compiling kernels at path : " << currentKernelPath << " in : " << currentIncludeDir << '\n';
	}

	const auto includeArg{ "-I " + currentIncludeDir };

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
		oroFunctions[record.kernelType] = OrochiUtils::getFunctionFromFile( device, currentKernelPath.c_str(), record.kernelName.c_str(), &opts );
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

	constexpr std::array<Kernel, 3UL> selectedKernels{ Kernel::COUNT, Kernel::SCAN_PARALLEL, Kernel::SORT };

	std::vector<int> sharedMemBytes( std::size( selectedKernels ) );
	std::transform( std::cbegin( selectedKernels ), std::cend( selectedKernels ), std::begin( sharedMemBytes ),
					[&]( const auto& kernel ) noexcept
					{
						int sharedMemory{ 0 };
						const auto func = oroFunctions[kernel];
						oroFuncGetAttribute( &sharedMemory, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );

						return sharedMemory;
					} );

	const auto maxSharedMemory = std::max_element( std::cbegin( sharedMemBytes ), std::cend( sharedMemBytes ) );
	const int occupancyFromLDS = ( maxSharedMemory != std::cend( sharedMemBytes ) && *maxSharedMemory > 0 ) ? props.sharedMemPerBlock / *maxSharedMemory : 1;

	const int occupancy = std::min( occupancyFromLDS, occupancyFromWarp );

	if( m_flags == Flag::LOG ) std::cout << "Occupancy: " << occupancy << '\n';

	return props.multiProcessorCount * occupancy;
}

RadixSort::u32 RadixSort::configure( oroDevice device, const std::string& kernelPath, const std::string& includeDir ) noexcept
{
	compileKernels( device, kernelPath, includeDir );
	const auto newWGsToExecute{ calculateWGsToExecute( device ) };

	if( newWGsToExecute != m_nWGsToExecute && selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_partialSum );
		OrochiUtils::free( m_isReady );
		OrochiUtils::malloc( m_partialSum, newWGsToExecute );
		OrochiUtils::malloc( m_isReady, newWGsToExecute );
		OrochiUtils::memset( m_isReady, false, newWGsToExecute * sizeof( bool ) );
	}

	m_nWGsToExecute = newWGsToExecute;
	return static_cast<u32>( BIN_SIZE * m_nWGsToExecute );
}
void RadixSort::setFlag( Flag flag ) noexcept { m_flags = flag; }

void RadixSort::sort( const KeyValueSoA src, const KeyValueSoA dst, int n, int startBit, int endBit, u32* tempBuffer ) noexcept
{
	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS_KV];
		const void* args[] = { &src.key, &src.value, &dst.key, &dst.value, &n, &startBit, &endBit };
		OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE );
		return;
	}

	auto* s{ &src };
	auto* d{ &dst };

	for( int i = startBit; i < endBit; i += N_RADIX )
	{
		sort1pass( *s, *d, n, i, i + std::min( N_RADIX, endBit - i ), (int*)tempBuffer );

		std::swap( s, d );
	}

	if( s == &src )
	{
		OrochiUtils::copyDtoD( dst.key, src.key, n );
		OrochiUtils::copyDtoD( dst.value, src.value, n );
	}
}

void RadixSort::sort( const u32* src, const u32* dst, int n, int startBit, int endBit, u32* tempBuffer ) noexcept
{
	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS];
		const void* args[] = { &src, &dst, &n, &startBit, &endBit };
		OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE );
		return;
	}

	auto* s{ &src };
	auto* d{ &dst };

	for( int i = startBit; i < endBit; i += N_RADIX )
	{
		sort1pass( *s, *d, n, i, i + std::min( N_RADIX, endBit - i ), (int*)tempBuffer );

		std::swap( s, d );
	}

	if( s == &src )
	{
		OrochiUtils::copyDtoD( dst, src, n );
	}
}

}; // namespace Oro
