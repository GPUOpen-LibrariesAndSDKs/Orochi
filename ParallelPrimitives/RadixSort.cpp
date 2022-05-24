#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSort.h>
#include <numeric>
#include <array>
#include <algorithm>
#include <Test/Stopwatch.h>

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

	constexpr auto ENABLE_PRINT{ false };

	if constexpr( ENABLE_PRINT )
	{
		for( int j = 0; j < nWGsToExecute; j++ )
		{
			for( int i = 0; i < Oro::BIN_SIZE; i++ )
			{
				printf( "%d, ", counts[j * Oro::BIN_SIZE + i] );
			}
			printf( "\n" );
		}
	}

	std::vector<int> offsets( Oro::BIN_SIZE * nWGsToExecute );
	std::exclusive_scan( std::cbegin( counts ), std::cend( counts ), std::begin( offsets ), 0 );

	OrochiUtils::copyHtoD( offsetsGpu, offsets.data(), Oro::BIN_SIZE * nWGsToExecute );
	OrochiUtils::waitForCompletion();
}


void printKernelInfo( oroFunction func )
{
	int a, b, c;
	oroFuncGetAttribute( &a, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
	oroFuncGetAttribute( &b, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
	oroFuncGetAttribute( &c, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
	printf( "vgpr : shared = %d : %d : %d\n", a, b, c );
}

} // namespace

namespace Oro
{

RadixSort::RadixSort()
{
	m_flags = (Flag)0;

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

void RadixSort::compileKernels( oroDevice device )
{
	constexpr auto kernelPath{ "../Test/ParallelPrimitives/RadixSortKernels.h" };

	printf( "compiling kernels ... \n" );

	std::vector<const char*> opts;
	//	opts.push_back( "--save-temps" );
	opts.push_back( "-I ../" );
	//	opts.push_back( "-G" );

	oroFunctions[Kernel::COUNT] = OrochiUtils::getFunctionFromFile( device, kernelPath, "CountKernel", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::COUNT] );

	oroFunctions[Kernel::COUNT_REF] = OrochiUtils::getFunctionFromFile( device, kernelPath, "CountKernelReference", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::COUNT_REF] );

	oroFunctions[Kernel::SCAN_SINGLE_WG] = OrochiUtils::getFunctionFromFile( device, kernelPath, "ParallelExclusiveScanSingleWG", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::SCAN_SINGLE_WG] );

	oroFunctions[Kernel::SCAN_PARALLEL] = OrochiUtils::getFunctionFromFile( device, kernelPath, "ParallelExclusiveScanAllWG", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::SCAN_PARALLEL] );

	oroFunctions[Kernel::SORT] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortKernel", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::SORT] );

	oroFunctions[Kernel::SORT_KV] = OrochiUtils::getFunctionFromFile(device, kernelPath, "SortKVKernel", &opts);
	if (m_flags & FLAG_LOG) printKernelInfo(oroFunctions[Kernel::SORT_KV]);

	oroFunctions[Kernel::SORT_SINGLE_PASS] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortSinglePassKernel", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::SORT_SINGLE_PASS] );

	oroFunctions[Kernel::SORT_SINGLE_PASS_KV] = OrochiUtils::getFunctionFromFile( device, kernelPath, "SortSinglePassKVKernel", &opts );
	if( m_flags & FLAG_LOG ) printKernelInfo( oroFunctions[Kernel::SORT_SINGLE_PASS_KV] );
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

	if( m_flags & FLAG_LOG ) printf( "Occupancy: %d\n", occupancy );

	return props.multiProcessorCount * occupancy;
}

void RadixSort::configure( oroDevice device, u32& tempBufferSizeOut )
{
	compileKernels( device );
	const auto newWGsToExecute = calculateWGsToExecute( device );

	if( newWGsToExecute != m_nWGsToExecute && selectedScanAlgo == ScanAlgo::SCAN_GPU_PARALLEL )
	{
		OrochiUtils::free( m_partialSum );
		OrochiUtils::free( m_isReady );
		OrochiUtils::malloc( m_partialSum, newWGsToExecute );
		OrochiUtils::malloc( m_isReady, newWGsToExecute );
		OrochiUtils::memset( m_isReady, false, newWGsToExecute * sizeof( bool ) );
	}

	m_nWGsToExecute = newWGsToExecute;
	tempBufferSizeOut = BIN_SIZE * m_nWGsToExecute;
}
void RadixSort::setFlag( Flag flag ) { m_flags = flag; }

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
