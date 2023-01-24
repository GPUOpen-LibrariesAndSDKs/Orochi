

template<class T>
void RadixSort::sort1pass( const T src, const T dst, int n, int startBit, int endBit, int* temps, oroStream stream ) noexcept
{
	constexpr bool reference = false;

	const u32* srcKey{ nullptr };
	const u32* dstKey{ nullptr };

	const u32* srcVal{ nullptr };
	const u32* dstVal{ nullptr };

	constexpr auto keyValuePairedEnabled{ std::is_same_v<T, KeyValueSoA> };

	if constexpr( keyValuePairedEnabled )
	{
		srcKey = src.key;
		dstKey = dst.key;

		srcVal = src.value;
		dstVal = dst.value;
	}
	else
	{
		static_assert( std::is_same_v<T, u32*> || std::is_same_v<T, const u32*> );
		srcKey = src;
		dstKey = dst;
	}

	// allocate temps
	// clear temps
	// count kernel
	// scan
	// sort

	const int nWIs = WG_SIZE * m_nWGsToExecute;
	int nItemsPerWI = ( n + ( nWIs - 1 ) ) / nWIs;

	// Adjust nItemsPerWI to be dividable by SORT_N_ITEMS_PER_WI.
	nItemsPerWI = ( std::ceil( static_cast<double>( nItemsPerWI ) / SORT_N_ITEMS_PER_WI ) ) * SORT_N_ITEMS_PER_WI;

	int nItemPerWG = nItemsPerWI * WG_SIZE;

	if( m_flags == Flag::LOG )
	{
		printf( "nWGs: %d\n", m_nWGsToExecute );
		printf( "nNItemsPerWI: %d\n", nItemsPerWI );
		printf( "nItemPerWG: %d\n", nItemPerWG );
	}

	float t[3] = { 0.f };
	{
		Stopwatch sw;
		sw.start();
		const auto func{ reference ? oroFunctions[Kernel::COUNT_REF] : oroFunctions[Kernel::COUNT] };
		const void* args[] = { &srcKey, &temps, &n, &nItemPerWG, &startBit, &m_nWGsToExecute };
		OrochiUtils::launch1D( func, COUNT_WG_SIZE * m_nWGsToExecute, args, COUNT_WG_SIZE, 0, stream );
#if defined( PROFILE )
		OrochiUtils::waitForCompletion( stream );
		sw.stop();
		t[0] = sw.getMs();
#endif
	}

	{
		Stopwatch sw;
		sw.start();
		switch( selectedScanAlgo )
		{
		case ScanAlgo::SCAN_CPU:
		{
			exclusiveScanCpu( temps, temps, m_nWGsToExecute, stream );
		}
		break;

		case ScanAlgo::SCAN_GPU_SINGLE_WG:
		{
			const void* args[] = { &temps, &temps, &m_nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_SINGLE_WG], WG_SIZE * m_nWGsToExecute, args, WG_SIZE, 0, stream );
		}
		break;

		case ScanAlgo::SCAN_GPU_PARALLEL:
		{
			const void* args[] = { &temps, &temps, arg_cast( m_partialSum.address() ), &m_isReady };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_PARALLEL], SCAN_WG_SIZE * m_nWGsToExecute, args, SCAN_WG_SIZE, 0, stream );
		}
		break;

		default:
			exclusiveScanCpu( temps, temps, m_nWGsToExecute, stream );
			break;
		}
#if defined( PROFILE )
		OrochiUtils::waitForCompletion( stream );
		sw.stop();
		t[1] = sw.getMs();
#endif
	}

	{
		Stopwatch sw;
		sw.start();

		if constexpr( keyValuePairedEnabled )
		{
			const void* args[] = { &srcKey, &srcVal, &dstKey, &dstVal, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT_KV], SORT_WG_SIZE * m_nWGsToExecute, args, SORT_WG_SIZE, 0, stream );
		}
		else
		{
			const void* args[] = { &srcKey, &dstKey, &temps, &n, &nItemsPerWI, &startBit, &m_nWGsToExecute };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT], SORT_WG_SIZE * m_nWGsToExecute, args, SORT_WG_SIZE, 0, stream );
		}

#if defined( PROFILE )
		OrochiUtils::waitForCompletion( stream );
		sw.stop();
		t[2] = sw.getMs();
#endif
	}
#if defined( PROFILE )
	printf( "%3.2f, %3.2f, %3.2f\n", t[0], t[1], t[2] );
#endif
}
