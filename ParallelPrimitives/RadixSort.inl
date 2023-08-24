

template<class T>
void RadixSort::sort1pass( const T src, const T dst, int n, int startBit, int endBit, oroStream stream ) noexcept
{
	constexpr auto enable_profile = false;

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

	const int nWIs = WG_SIZE * m_num_blocks_for_count;
	int nItemsPerWI = ( n + ( nWIs - 1 ) ) / nWIs;

	// Adjust nItemsPerWI to be dividable by SORT_N_ITEMS_PER_WI.
	nItemsPerWI = ( std::ceil( static_cast<double>( nItemsPerWI ) / SORT_N_ITEMS_PER_WI ) ) * SORT_N_ITEMS_PER_WI;

	const int nItemPerWG = nItemsPerWI * WG_SIZE;

	if( m_flags == Flag::LOG )
	{
		printf( "num_blocks_for_count: %d\n", m_num_blocks_for_count );
		printf( "num_blocks_for_scan: %d\n", m_num_blocks_for_scan );
		printf( "nNItemsPerWI: %d\n", nItemsPerWI );
		printf( "nItemPerWG: %d\n", nItemPerWG );

		std::cout << "Input size n: " << n << std::endl;
	}

	float t[3] = { 0.f };

	{
		const auto num_total_thread_for_count = COUNT_WG_SIZE * m_num_blocks_for_count;

		Stopwatch sw;
		sw.start();
		const auto func{ oroFunctions[Kernel::COUNT] };
		const void* args[] = { &srcKey, arg_cast( m_tmp_buffer.address() ), &n, &nItemPerWG, &startBit, &m_num_blocks_for_count };
		OrochiUtils::launch1D( func, num_total_thread_for_count, args, COUNT_WG_SIZE, 0, stream );

		if constexpr( enable_profile )
		{
			OrochiUtils::waitForCompletion( stream );
			sw.stop();
			t[0] = sw.getMs();
		}
	}

	{
		Stopwatch sw;
		sw.start();
		switch( selectedScanAlgo )
		{
		case ScanAlgo::SCAN_CPU:
		{
			exclusiveScanCpu( m_tmp_buffer, m_tmp_buffer, m_num_blocks_for_count, stream );
		}
		break;

		case ScanAlgo::SCAN_GPU_SINGLE_WG:
		{
			const void* args[] = { arg_cast( m_tmp_buffer.address() ), arg_cast( m_tmp_buffer.address() ), &m_num_blocks_for_count };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_SINGLE_WG], WG_SIZE * m_num_blocks_for_count, args, WG_SIZE, 0, stream );
		}
		break;

		case ScanAlgo::SCAN_GPU_PARALLEL:
		{
			const auto num_total_thread_for_scan = SCAN_WG_SIZE * m_num_blocks_for_scan;

			const void* args[] = { arg_cast( m_tmp_buffer.address() ), arg_cast( m_tmp_buffer.address() ), arg_cast( m_partial_sum.address() ), arg_cast( m_is_ready.address() ) };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_PARALLEL], num_total_thread_for_scan, args, SCAN_WG_SIZE, 0, stream );
		}
		break;

		default:
			exclusiveScanCpu( m_tmp_buffer, m_tmp_buffer, m_num_blocks_for_count, stream );
			break;
		}

		if constexpr( enable_profile )
		{
			OrochiUtils::waitForCompletion( stream );
			sw.stop();
			t[1] = sw.getMs();
		}
	}

	{
		Stopwatch sw;
		sw.start();

		const auto num_blocks_for_sort = m_num_blocks_for_count;
		const auto num_total_thread_for_sort = SORT_WG_SIZE * num_blocks_for_sort;

		if constexpr( keyValuePairedEnabled )
		{
			const void* args[] = { &srcKey, &srcVal, &dstKey, &dstVal, arg_cast( m_tmp_buffer.address() ), &n, &nItemsPerWI, &startBit, &num_blocks_for_sort };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT_KV], num_total_thread_for_sort, args, SORT_WG_SIZE, 0, stream );
		}
		else
		{
			const void* args[] = { &srcKey, &dstKey, arg_cast( m_tmp_buffer.address() ), &n, &nItemsPerWI, &startBit, &num_blocks_for_sort };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT], num_total_thread_for_sort, args, SORT_WG_SIZE, 0, stream );
		}

		if constexpr( enable_profile )
		{
			OrochiUtils::waitForCompletion( stream );
			sw.stop();
			t[2] = sw.getMs();
		}
	}

	if constexpr( enable_profile )
	{
		printf( "%3.2f, %3.2f, %3.2f\n", t[0], t[1], t[2] );
	}
}
