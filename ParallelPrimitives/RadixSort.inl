

namespace
{

struct Empty
{
};

/// @brief Call the callable and measure the elapsed time using the Stopwatch.
/// @tparam CallableType The type of the callable to be invoked in this function.
/// @tparam RecordType The type of the object that stores the recorded times.
/// @tparam enable_profile The elapsed time will be recorded if this is set to True.
/// @param callable The callable object to be called.
/// @param time_record The object that stores the recorded times.
/// @param index The index indicates where to store the elapsed time in @c time_record
/// @param stream The GPU stream
template<bool enable_profile, typename CallableType, typename RecordType>
constexpr void execute( CallableType&& callable, RecordType& time_record, const int index, const oroStream stream ) noexcept
{
	using TimerType = std::conditional_t<enable_profile, Stopwatch, Empty>;

	TimerType stopwatch;

	if constexpr( enable_profile )
	{
		stopwatch.start();
	}

	std::invoke( std::forward<CallableType>( callable ) );

	if constexpr( enable_profile )
	{
		OrochiUtils::waitForCompletion( stream );
		stopwatch.stop();
		time_record[index] = stopwatch.getMs();
	}
}
} // namespace

template<class T>
void RadixSort::sort1pass( const T src, const T dst, int n, int startBit, int endBit, oroStream stream ) noexcept
{
	static constexpr auto enable_profile = false;

	const u32* srcKey{ nullptr };
	const u32* dstKey{ nullptr };

	const u32* srcVal{ nullptr };
	const u32* dstVal{ nullptr };

	static constexpr auto enable_key_value_pair_sorting{ std::is_same_v<T, KeyValueSoA> };

	if constexpr( enable_key_value_pair_sorting )
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

	const int nItemPerWG = ( n + m_num_blocks_for_count - 1 ) / m_num_blocks_for_count;

	// Timer records

	using RecordType = std::conditional_t<enable_profile, std::vector<float>, Empty>;
	RecordType t;

	if constexpr( enable_profile )
	{
		t.resize( 3 );
	}

	const auto launch_count_kernel = [&]() noexcept
	{
		const auto num_total_thread_for_count = m_num_threads_per_block_for_count * m_num_blocks_for_count;

		const auto func{ oroFunctions[Kernel::COUNT] };
		const void* args[] = { &srcKey, arg_cast( m_tmp_buffer.address() ), &n, &nItemPerWG, &startBit, &m_num_blocks_for_count };
		OrochiUtils::launch1D( func, num_total_thread_for_count, args, m_num_threads_per_block_for_count, 0, stream );
	};

	execute<enable_profile>( launch_count_kernel, t, 0, stream );

	const auto launch_scan_kernel = [&]() noexcept
	{
		switch( selectedScanAlgo )
		{
		case ScanAlgo::SCAN_CPU:
		{
			exclusiveScanCpu( m_tmp_buffer, m_tmp_buffer );
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
			const auto num_total_thread_for_scan = m_num_threads_per_block_for_scan * m_num_blocks_for_scan;

			const void* args[] = { arg_cast( m_tmp_buffer.address() ), arg_cast( m_tmp_buffer.address() ), arg_cast( m_partial_sum.address() ), arg_cast( m_is_ready.address() ) };
			OrochiUtils::launch1D( oroFunctions[Kernel::SCAN_PARALLEL], num_total_thread_for_scan, args, m_num_threads_per_block_for_scan, 0, stream );
		}
		break;

		default:
			exclusiveScanCpu( m_tmp_buffer, m_tmp_buffer );
			break;
		}
	};

	execute<enable_profile>( launch_scan_kernel, t, 1, stream );

	const auto launch_sort_kernel = [&]() noexcept
	{
		const auto num_blocks_for_sort = m_num_blocks_for_count;
		const auto num_total_thread_for_sort = m_num_threads_per_block_for_sort * num_blocks_for_sort;
		const auto num_items_per_block = nItemPerWG;

		if constexpr( enable_key_value_pair_sorting )
		{
			const void* args[] = { &srcKey, &srcVal, &dstKey, &dstVal, arg_cast( m_tmp_buffer.address() ), &n, &num_items_per_block, &startBit, &num_blocks_for_sort };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT_KV], num_total_thread_for_sort, args, m_num_threads_per_block_for_sort, 0, stream );
		}
		else
		{
			const void* args[] = { &srcKey, &dstKey, arg_cast( m_tmp_buffer.address() ), &n, &num_items_per_block, &startBit, &num_blocks_for_sort };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT], num_total_thread_for_sort, args, m_num_threads_per_block_for_sort, 0, stream );
		}
	};

	execute<enable_profile>( launch_sort_kernel, t, 2, stream );

	if constexpr( enable_profile )
	{
		printf( "%3.2f, %3.2f, %3.2f\n", t[0], t[1], t[2] );
	}
}
