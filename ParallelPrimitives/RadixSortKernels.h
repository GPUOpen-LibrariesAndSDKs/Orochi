#include <ParallelPrimitives/RadixSortConfigs.h>
#define LDS_BARRIER __syncthreads()

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
#define ITS 1
#endif

namespace
{

using namespace Oro;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
} // namespace

template<typename T>
__device__ T ldsScanExclusive( T* lds, int width )
{
	__shared__ T sum;

	int offset = 1;

	for( int d = width >> 1; d > 0; d >>= 1 )
	{

		if( threadIdx.x < d )
		{
			const int firstInputIndex = offset * ( 2 * threadIdx.x + 1 ) - 1;
			const int secondInputIndex = offset * ( 2 * threadIdx.x + 2 ) - 1;

			lds[secondInputIndex] += lds[firstInputIndex];
		}
		LDS_BARRIER;

		offset *= 2;
	}

	LDS_BARRIER;

	if( threadIdx.x == 0 )
	{
		sum = lds[width - 1];
		__threadfence_block();

		lds[width - 1] = 0;
		__threadfence_block();
	}

	for( int d = 1; d < width; d *= 2 )
	{
		offset >>= 1;

		if( threadIdx.x < d )
		{
			const int firstInputIndex = offset * ( 2 * threadIdx.x + 1 ) - 1;
			const int secondInputIndex = offset * ( 2 * threadIdx.x + 2 ) - 1;

			const T t = lds[firstInputIndex];
			lds[firstInputIndex] = lds[secondInputIndex];
			lds[secondInputIndex] += t;
		}
		LDS_BARRIER;
	}

	LDS_BARRIER;

	return sum;
}
//========================

__device__ void localSort4bitMultiRef( int* keys, u32* ldsKeys, const int START_BIT )
{
	__shared__ u32 ldsTemp[WG_SIZE + 1][N_BINS_4BIT];

	for( int i = 0; i < N_BINS_4BIT; i++ )
	{
		ldsTemp[threadIdx.x][i] = 0;
	}
	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		ldsTemp[threadIdx.x][in4bit] += 1;
	}

	LDS_BARRIER;

	if( threadIdx.x < N_BINS_4BIT ) // 16 scans, pack 4 scans into 1 to make 4 parallel scans
	{
		int sum = 0;
		for( int i = 0; i < WG_SIZE; i++ )
		{
			int t = ldsTemp[i][threadIdx.x];
			ldsTemp[i][threadIdx.x] = sum;
			sum += t;
		}
		ldsTemp[WG_SIZE][threadIdx.x] = sum;
	}
	LDS_BARRIER;
	if( threadIdx.x == 0 ) // todo parallel scan
	{
		int sum = 0;
		for( int i = 0; i < N_BINS_4BIT; i++ )
		{
			int t = ldsTemp[WG_SIZE][i];
			ldsTemp[WG_SIZE][i] = sum;
			sum += t;
		}
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		int offset = ldsTemp[WG_SIZE][in4bit];
		int rank = ldsTemp[threadIdx.x][in4bit]++;

		ldsKeys[offset + rank] = keys[i];
	}
	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

template<int N_ITEMS_PER_WI, int EXEC_WIDTH, bool KEY_VALUE_PAIR = true>
__device__ void localSort4bitMulti( int* keys, u32* ldsKeys, int* values, u32* ldsValues, const int START_BIT )
{
	__shared__ union
	{
		u16 m_unpacked[EXEC_WIDTH + 1][N_BINS_PACKED_4BIT][N_BINS_PACK_FACTOR];
		u64 m_packed[EXEC_WIDTH + 1][N_BINS_PACKED_4BIT];
	} lds;

	__shared__ u64 ldsTemp[EXEC_WIDTH];

	for( int i = 0; i < N_BINS_PACKED_4BIT; ++i )
	{
		lds.m_packed[threadIdx.x][i] = 0UL;
	}

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		const int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		const int packIdx = in4bit / N_BINS_PACK_FACTOR;
		const int idx = in4bit % N_BINS_PACK_FACTOR;
		lds.m_unpacked[threadIdx.x][packIdx][idx] += 1;
	}

	LDS_BARRIER;

	for( int ii = 0; ii < N_BINS_PACKED_4BIT; ++ii )
	{
		ldsTemp[threadIdx.x] = lds.m_packed[threadIdx.x][ii];
		LDS_BARRIER;
		const u64 sum = ldsScanExclusive( ldsTemp, EXEC_WIDTH );
		LDS_BARRIER;
		lds.m_packed[threadIdx.x][ii] = ldsTemp[threadIdx.x];

		if( threadIdx.x == 0 ) lds.m_packed[EXEC_WIDTH][ii] = sum;
	}

	LDS_BARRIER;

	auto* tmp = &lds.m_unpacked[EXEC_WIDTH][0][0];
	ldsScanExclusive( tmp, N_BINS_PACKED_4BIT * N_BINS_PACK_FACTOR );

	LDS_BARRIER;

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		const int in4bit = ( keys[i] >> START_BIT ) & 0xf;
		const int packIdx = in4bit / N_BINS_PACK_FACTOR;
		const int idx = in4bit % N_BINS_PACK_FACTOR;
		const int offset = lds.m_unpacked[EXEC_WIDTH][packIdx][idx];
		const int rank = lds.m_unpacked[threadIdx.x][packIdx][idx]++;

		ldsKeys[offset + rank] = keys[i];

		if constexpr( KEY_VALUE_PAIR )
		{
			ldsValues[offset + rank] = values[i];
		}
	}
	LDS_BARRIER;

	for( int i = 0; i < N_ITEMS_PER_WI; ++i )
	{
		keys[i] = ldsKeys[threadIdx.x * N_ITEMS_PER_WI + i];

		if constexpr( KEY_VALUE_PAIR )
		{
			values[i] = ldsValues[threadIdx.x * N_ITEMS_PER_WI + i];
		}
	}
}

template<bool KEY_VALUE_PAIR>
__device__ void SortSinglePass( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int gN, const int START_BIT, const int END_BIT )
{
	if( blockIdx.x > 0 )
	{
		return;
	}

	__shared__ u32 ldsKeys[SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI];
	__shared__ u32 ldsValues[KEY_VALUE_PAIR ? SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI : 1];

	int keys[SINGLE_SORT_N_ITEMS_PER_WI] = { 0 };
	int values[KEY_VALUE_PAIR ? SINGLE_SORT_N_ITEMS_PER_WI : 1] = { 0 };

	for( int i = 0; i < SINGLE_SORT_N_ITEMS_PER_WI; i++ )
	{
		const int idx = threadIdx.x * SINGLE_SORT_N_ITEMS_PER_WI + i;
		keys[i] = ( idx < gN ) ? gSrcKey[idx] : 0xffffffff;
		ldsKeys[idx] = keys[i];

		if constexpr( KEY_VALUE_PAIR )
		{
			values[i] = ( idx < gN ) ? gSrcVal[idx] : 0xffffffff;
			ldsValues[idx] = values[i];
		}
	}

	LDS_BARRIER;

	for( int bit = START_BIT; bit < END_BIT; bit += N_RADIX )
	{
		localSort4bitMulti<SINGLE_SORT_N_ITEMS_PER_WI, SINGLE_SORT_WG_SIZE, KEY_VALUE_PAIR>( keys, ldsKeys, values, ldsValues, bit );
		localSort4bitMulti<SINGLE_SORT_N_ITEMS_PER_WI, SINGLE_SORT_WG_SIZE, KEY_VALUE_PAIR>( keys, ldsKeys, values, ldsValues, bit + 4 );
	}

	for( int i = 0; i < SINGLE_SORT_N_ITEMS_PER_WI; i++ )
	{
		const int idx = threadIdx.x * SINGLE_SORT_N_ITEMS_PER_WI + i;
		if( idx < gN )
		{
			gDstKey[idx] = keys[i];

			if constexpr( KEY_VALUE_PAIR )
			{
				gDstVal[idx] = values[i];
			}
		}
	}
}

extern "C" __global__ void SortSinglePassKernel( int* gSrcKey, int* gDstKey, int gN, const int START_BIT, const int END_BIT ) { SortSinglePass<false>( gSrcKey, nullptr, gDstKey, nullptr, gN, START_BIT, END_BIT ); }

extern "C" __global__ void SortSinglePassKVKernel( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int gN, const int START_BIT, const int END_BIT ) { SortSinglePass<true>( gSrcKey, gSrcVal, gDstKey, gDstVal, gN, START_BIT, END_BIT ); }

using RADIX_SORT_KEY_TYPE = u32;
using RADIX_SORT_VALUE_TYPE = u32;

#if defined( DESCENDING_ORDER )
constexpr u32 ORDER_MASK_32 = 0xFFFFFFFF;
constexpr u64 ORDER_MASK_64 = 0xFFFFFFFFFFFFFFFFllu;
#else
constexpr u32 ORDER_MASK_32 = 0;
constexpr u64 ORDER_MASK_64 = 0llu;
#endif

__device__ constexpr u32 div_round_up( u32 val, u32 divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }

template<int NElement, int NThread, class T>
__device__ void clearShared( T* sMem, T value )
{
	for( int i = 0; i < NElement; i += NThread )
	{
		if( i < NElement )
		{
			sMem[i + threadIdx.x] = value;
		}
	}
}

__device__ inline u32 getKeyBits( u32 x ) { return x ^ ORDER_MASK_32; }
__device__ inline u64 getKeyBits( u64 x ) { return x ^ ORDER_MASK_64; }
__device__ inline u32 extractDigit( u32 x, u32 bitLocation ) { return ( x >> bitLocation ) & RADIX_MASK; }
__device__ inline u32 extractDigit( u64 x, u32 bitLocation ) { return (u32)( ( x >> bitLocation ) & RADIX_MASK ); }

template <class T>
__device__ inline T scanExclusive( T prefix, T* sMemIO, int nElement )
{
	// assert(nElement <= nThreads)
	bool active = threadIdx.x < nElement;
	T value = active ? sMemIO[threadIdx.x] : 0;
	T x = value;

	for( u32 offset = 1; offset < nElement; offset <<= 1 )
	{
		if( active && offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		if( active )
		{
			sMemIO[threadIdx.x] = x;
		}

		__syncthreads();
	}

	T sum = sMemIO[nElement - 1];

	__syncthreads();

	if( active )
	{
		sMemIO[threadIdx.x] = x + prefix - value;
	}

	__syncthreads();

	return sum;
}

extern "C" __global__ void gHistogram( RADIX_SORT_KEY_TYPE* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
	__shared__ u32 localCounters[sizeof( RADIX_SORT_KEY_TYPE )][BIN_SIZE];

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += GHISTOGRAM_THREADS_PER_BLOCK )
		{
			localCounters[i][j] = 0;
		}
	}

	u32 numberOfBlocks = div_round_up( numberOfInputs, GHISTOGRAM_ITEM_PER_BLOCK );
	__shared__ u32 iBlock;
	for(;;)
	{
		if( threadIdx.x == 0 )
		{
			iBlock = atomicInc( counter, 0xFFFFFFFF );
		}

		__syncthreads();

		if( numberOfBlocks <= iBlock )
			break;
    
		for( int j = 0; j < GHISTOGRAM_ITEMS_PER_THREAD; j++ )
		{
			u32 itemIndex = iBlock * GHISTOGRAM_ITEM_PER_BLOCK + threadIdx.x * GHISTOGRAM_ITEMS_PER_THREAD + j;
			if( itemIndex < numberOfInputs )
			{
				auto item = inputs[itemIndex];
				for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
				{
					u32 bitLocation = startBits + i * N_RADIX;
					u32 bits = extractDigit( getKeyBits( item ), bitLocation );
					atomicInc( &localCounters[i][bits], 0xFFFFFFFF );
				}
			}
		}

		__syncthreads();
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		scanExclusive<u32>( 0, &localCounters[i][0], BIN_SIZE );
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += GHISTOGRAM_THREADS_PER_BLOCK )
		{
			atomicAdd( &gpSumBuffer[BIN_SIZE * i + j], localCounters[i][j] );
		}
	}
}

template <bool keyPair>
__device__ __forceinline__ void onesweep_reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* gpSumBuffer,
												  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	__shared__ u32 pSum[BIN_SIZE];

	struct SMem
	{
		struct Phase1
		{
			u16 blockHistogram[BIN_SIZE];
			u16 lpSum[BIN_SIZE * REORDER_NUMBER_OF_WARPS];
		};
		struct Phase2
		{
			RADIX_SORT_KEY_TYPE elements[RADIX_SORT_BLOCK_SIZE];
		};
		struct Phase3
		{
			RADIX_SORT_VALUE_TYPE elements[RADIX_SORT_BLOCK_SIZE];
			u8 buckets[RADIX_SORT_BLOCK_SIZE];
		};

		union
		{
			Phase1 phase1;
			Phase2 phase2;
			Phase3 phase3;
		} u;
	};
	__shared__ SMem smem;

	u32 bitLocation = startBits + N_RADIX * iteration;
	u32 blockIndex = blockIdx.x;
	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	clearShared<BIN_SIZE * REORDER_NUMBER_OF_WARPS, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u16>( smem.u.phase1.lpSum, 0 );

	__syncthreads();

	RADIX_SORT_KEY_TYPE keys[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	u32 warpOffsets[REORDER_NUMBER_OF_ITEM_PER_THREAD];

	int warp = threadIdx.x / WARP_SIZE;
	int lane = threadIdx.x % WARP_SIZE;

	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		if( itemIndex < numberOfInputs )
		{
			keys[k] = inputKeys[itemIndex];
		}
	}
	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );

		// check the attendees
		u32 broThreads =
#if defined( ITS )
			__ballot_sync( 0xFFFFFFFF,
#else
			__ballot(
#endif
						   itemIndex < numberOfInputs );

		for( int j = 0; j < N_RADIX; ++j )
		{
			u32 bit = ( bucketIndex >> j ) & 0x1;
			u32 difference = ( 0xFFFFFFFF * bit ) ^
#if defined( ITS )
								__ballot_sync( 0xFFFFFFFF, bit != 0 );
#else
								__ballot( bit != 0 );
#endif
			broThreads &= ~difference;
		}

		u32 lowerMask = ( 1u << lane ) - 1;
		auto digitCount = smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		warpOffsets[k] = digitCount + __popc( broThreads & lowerMask );
		
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
		u32 leaderIdx = __ffs( broThreads ) - 1;
		if( lane == leaderIdx )
		{
			smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] = digitCount + __popc( broThreads );
		}
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
	}

	__syncthreads();

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = 0;
		for( int warp = 0; warp < REORDER_NUMBER_OF_WARPS; warp++ )
		{
			s += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		}
		smem.u.phase1.blockHistogram[bucketIndex] = s;
	}

	struct ParitionID
	{
		u64 value : 32;
		u64 block : 30;
		u64 flag : 2;
	};
	auto asPartition = []( u64 x )
	{
		ParitionID pa;
		memcpy( &pa, &x, sizeof( ParitionID ) );
		return pa;
	};
	auto asU64 = []( ParitionID pa )
	{
		u64 x;
		memcpy( &x, &pa, sizeof( u64 ) );
		return x;
	};

	if( threadIdx.x == 0 && LOOKBACK_TABLE_SIZE <= blockIndex )
	{
		// Wait until blockIndex < tail - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE <= blockIndex )
			;
	}
	__syncthreads();

	for( int i = threadIdx.x; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[i];
		int pIndex = BIN_SIZE * ( blockIndex % LOOKBACK_TABLE_SIZE ) + i;

		{
			ParitionID pa;
			pa.value = s;
			pa.block = blockIndex;
			pa.flag = 1;
			lookBackBuffer[pIndex] = asU64( pa );
		}

		u32 gp = gpSumBuffer[iteration * BIN_SIZE + i];

		u32 p = 0;

		for( int iBlock = (int)blockIndex - 1; 0 <= iBlock; iBlock-- )
		{
			int lookbackIndex = BIN_SIZE * ( iBlock % LOOKBACK_TABLE_SIZE ) + i;
			ParitionID pa;

			// when you reach to the maximum, flag must be 2. flagRequire = 0b10
			// Otherwise, flag can be 1 or 2 flagRequire = 0b11
			int flagRequire = MAX_LOOK_BACK == blockIndex - iBlock ? 2 : 3;

			do
			{
				pa = asPartition( lookBackBuffer[lookbackIndex] );
			} while( ( pa.flag & flagRequire ) == 0 || pa.block != iBlock );

			u32 value = pa.value;
			p += value;
			if( pa.flag == 2 )
			{
				break;
			}
		}

		ParitionID pa;
		pa.value = p + s;
		pa.block = blockIndex;
		pa.flag = 2;
		lookBackBuffer[pIndex] = asU64( pa );

		// complete global output location
		u32 globalOutput = gp + p;
		pSum[i] = globalOutput;
	}

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) != ( blockIndex & TAIL_MASK ) )
			;

		atomicInc( tailIterator, numberOfBlocks - 1 /* after the vary last item, it will be zero */ );
	}

	__syncthreads();

	u32 prefix = 0;
	for( int i = 0; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += scanExclusive<u16>( prefix, smem.u.phase1.blockHistogram + i, min( REORDER_NUMBER_OF_THREADS_PER_BLOCK, BIN_SIZE ) );
	}

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[bucketIndex];

		pSum[bucketIndex] -= s; // pre-substruct to avoid pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex] to calculate destinations

		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			int index = bucketIndex * REORDER_NUMBER_OF_WARPS + w;
			u32 n = smem.u.phase1.lpSum[index];
			smem.u.phase1.lpSum[index] = s;
			s += n;
		}
	}

	__syncthreads();

	for( int k = 0; k < REORDER_NUMBER_OF_ITEM_PER_THREAD; k++ )
	{
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		warpOffsets[k] += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
	}

	__syncthreads();

	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		if( itemIndex < numberOfInputs )
		{
			smem.u.phase2.elements[warpOffsets[k]] = keys[k];
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = smem.u.phase2.elements[i];
			u32 bucketIndex = extractDigit( getKeyBits( item ), bitLocation );

			// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
			u32 dstIndex = pSum[bucketIndex] + i;
			outputKeys[dstIndex] = item;
		}
	}

	if constexpr( keyPair )
	{
		__syncthreads();

		for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
			u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
			if( itemIndex < numberOfInputs )
			{
				smem.u.phase3.elements[warpOffsets[k]] = inputValues[itemIndex];
				smem.u.phase3.buckets[warpOffsets[k]] = bucketIndex;
			}
		}

		__syncthreads();

		for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
			if( itemIndex < numberOfInputs )
			{
				auto item       = smem.u.phase3.elements[i];
				u32 bucketIndex = smem.u.phase3.buckets[i];

				// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
				u32 dstIndex = pSum[bucketIndex] + i;
				outputValues[dstIndex] = item;
			}
		}
	}
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKey64( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, u32 numberOfInputs, u32* gpSumBuffer, volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits,
												  u32 iteration )
{
	onesweep_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKeyPair64( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues,
																											   u32 numberOfInputs,
																								   u32* gpSumBuffer,
													  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	onesweep_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}