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


constexpr auto KEY_IS_16BYTE_ALIGNED = true;

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
__device__ __forceinline__ u32 u32min( u32 x, u32 y ) { return ( y < x ) ? y : x; }

template<int NThreads>
__device__ inline u32 prefixSumExclusive( u32 prefix, u32* sMemIO )
{
	u32 value = sMemIO[threadIdx.x];

	for( u32 offset = 1; offset < NThreads; offset <<= 1 )
	{
		u32 x = sMemIO[threadIdx.x];

		if( offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		sMemIO[threadIdx.x] = x;

		__syncthreads();
	}
	u32 sum = sMemIO[NThreads - 1];

	__syncthreads();

	sMemIO[threadIdx.x] += prefix - value;

	__syncthreads();

	return sum;
}

extern "C" __global__ void gHistogram( RADIX_SORT_KEY_TYPE* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
	__shared__ u32 localCounters[sizeof( RADIX_SORT_KEY_TYPE )][256];

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < 256; j += GHISTOGRAM_THREADS_PER_BLOCK )
		{
			localCounters[i][j] = 0;
		}
	}

	__syncthreads();

	u32 numberOfBlocks = div_round_up( numberOfInputs, GHISTOGRAM_ITEM_PER_BLOCK );
	__shared__ u32 iBlock;
	if( threadIdx.x == 0 )
	{
		iBlock = atomicInc( counter, 0xFFFFFFFF );
	}

	__syncthreads();

	bool hasData = false;

	while( iBlock < numberOfBlocks )
	{
		hasData = true;

		if( KEY_IS_16BYTE_ALIGNED && ( iBlock + 1 ) * GHISTOGRAM_ITEM_PER_BLOCK <= numberOfInputs )
		{
			for( int i = 0; i < GHISTOGRAM_ITEM_PER_BLOCK; i += GHISTOGRAM_THREADS_PER_BLOCK * 4 )
			{
				u32 itemIndex = iBlock * GHISTOGRAM_ITEM_PER_BLOCK + i + threadIdx.x * 4;
				struct alignas( 16 ) Key4
				{
					RADIX_SORT_KEY_TYPE xs[4];
				};
				Key4 key4 = *(Key4*)&inputs[itemIndex];
				for( int k = 0; k < 4; k++ )
				{
					auto item = key4.xs[k];
					for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
					{
						u32 bitLocation = startBits + i * 8;
						u32 bits = extractDigit( getKeyBits( item ), bitLocation );
						atomicInc( &localCounters[i][bits], 0xFFFFFFFF );
					}
				}
			}
		}
		else
		{
			for( int i = threadIdx.x; i < GHISTOGRAM_ITEM_PER_BLOCK; i += GHISTOGRAM_THREADS_PER_BLOCK )
			{
				u32 itemIndex = iBlock * GHISTOGRAM_ITEM_PER_BLOCK + i;
				if( itemIndex < numberOfInputs )
				{
					auto item = inputs[itemIndex];
					for( int j = 0; j < sizeof( RADIX_SORT_KEY_TYPE ); j++ )
					{
						u32 bitLocation = startBits + j * 8;
						u32 bits = extractDigit( getKeyBits( item ), bitLocation );
						atomicInc( &localCounters[j][bits], 0xFFFFFFFF );
					}
				}
			}
		}
		__syncthreads();

		if( threadIdx.x == 0 )
		{
			iBlock = atomicInc( counter, 0xFFFFFFFF );
		}

		__syncthreads();
	}

	if( hasData )
	{
		__syncthreads();

		for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
		{
			for( int j = threadIdx.x; j < BIN_SIZE; j += GHISTOGRAM_THREADS_PER_BLOCK )
			{
				atomicAdd( &gpSumBuffer[BIN_SIZE * i + j], localCounters[i][j] );
			}
		}
	}
}

extern "C" __global__ void gPrefixSum( u32* gpSumBuffer )
{
	__shared__ u32 smem[BIN_SIZE];

	smem[threadIdx.x] = gpSumBuffer[blockIdx.x * BIN_SIZE + threadIdx.x];

	__syncthreads();

	prefixSumExclusive<BIN_SIZE>( 0, smem );

	gpSumBuffer[blockIdx.x * BIN_SIZE + threadIdx.x] = smem[threadIdx.x];
}

template <bool keyPair>
__device__ __forceinline__ void onesweep_reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* gpSumBuffer,
												  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	struct ElementLocation
	{
		u32 localSrcIndex : 12;
		u32 localOffset : 12;
		u32 bucket : 8;
	};

	__shared__ u32 blockHistogram[BIN_SIZE];
	__shared__ u32 lpSum[BIN_SIZE * REORDER_NUMBER_OF_WARPS];
	__shared__ u32 pSum[BIN_SIZE];
	__shared__ ElementLocation elementLocations[RADIX_SORT_BLOCK_SIZE];

	//__shared__ u32 localPrefixSum[BIN_SIZE];
	//__shared__ u32 counters[BIN_SIZE];
	//__shared__ ElementLocation elementLocations[RADIX_SORT_BLOCK_SIZE];
	//__shared__ u8 elementBuckets[RADIX_SORT_BLOCK_SIZE];
	//__shared__ u32 matchMasks[REORDER_NUMBER_OF_WARPS][BIN_SIZE];

	u32 bitLocation = startBits + 8 * iteration;
	u32 blockIndex = blockIdx.x;
	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	// clearShared<BIN_SIZE, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u32>( localPrefixSum, 0 );
	// clearShared<BIN_SIZE, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u32>( counters, 0 );

	//for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
	//{
	//	for( int i = threadIdx.x; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	//	{
	//		matchMasks[w][i] = 0;
	//	}
	//}

	clearShared<BIN_SIZE, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u32>( blockHistogram, 0 );
	clearShared<BIN_SIZE * REORDER_NUMBER_OF_WARPS, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u32>( lpSum, 0 );

	__syncthreads();

	u8  bucketIndices[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	u32 warpOffsets[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	// u32 bros[REORDER_NUMBER_OF_ITEM_PER_THREAD];

	int warp = threadIdx.x / 32;
	int lane = threadIdx.x % 32;
	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += 32, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;

		u32 bucketIndex = 0;
		if( itemIndex < numberOfInputs )
		{
			auto item = inputKeys[itemIndex];
			bucketIndex = extractDigit( getKeyBits( item ), bitLocation );
		}
		bucketIndices[k] = bucketIndex;

		int nNoneActiveItems = 32 - u32min( numberOfInputs - ( itemIndex - lane ), 32 ); // 0 - 32
		u32 broThreads = 0xFFFFFFFF >> nNoneActiveItems;

		for( int j = 0; j < 8; ++j )
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
		// bros[k] = broThreads;
		int laneIndex = threadIdx.x % 32;
		u32 lowerMask = ( 1u << laneIndex ) - 1;

		if( itemIndex < numberOfInputs )
		{
			warpOffsets[k] = lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] + __popc( broThreads & lowerMask );
		}

		bool leader = ( broThreads & lowerMask ) == 0;
		if( itemIndex < numberOfInputs && leader )
		{
			u32 n = __popc( broThreads );
			atomicAdd( &blockHistogram[bucketIndex], n );
			lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] += n;
		}
	}

	__syncthreads();

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
		u32 mustBeDone = blockIndex - LOOKBACK_TABLE_SIZE + MAX_LOOK_BACK;
		while( ( atomicAdd( tailIterator, 0 ) >> TAIL_BITS ) * TAIL_COUNT <= mustBeDone )
			;
	}
	__syncthreads();

	for( int i = threadIdx.x; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		//u32 s = localPrefixSum[i];
		u32 s = blockHistogram[i];
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


	u32 prefix = 0;
	for( int i = 0; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += prefixSumExclusive<REORDER_NUMBER_OF_THREADS_PER_BLOCK>( prefix, &blockHistogram[i] );
	}
	{
		int bucketIndex = threadIdx.x;
		u32 s = blockHistogram[bucketIndex];
		for( int warp = 0; warp < REORDER_NUMBER_OF_WARPS; warp++ )
		{
			int index = bucketIndex * REORDER_NUMBER_OF_WARPS + warp;
			u32 n = lpSum[index];
			lpSum[index] = s;
			s += n;
		}
	}
	// printf( "[%d] %d\n", threadIdx.x, blockHistogram[threadIdx.x] );

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		while( ( atomicAdd( tailIterator, 0 ) >> TAIL_BITS ) != blockIndex / TAIL_COUNT )
			;

		atomicInc( tailIterator, numberOfBlocks - 1 /* after the vary last item, it will be zero */ );
	}

	__syncthreads();

	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += 32, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
		u32 bucketIndex = bucketIndices[k];

		if( itemIndex < numberOfInputs )
		{
			u32 localBase = lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
			// u32 to = localBase + __popc( broThreads & lowerMask );
			u32 to = localBase + warpOffsets[k];

			ElementLocation el;
			el.localSrcIndex = itemIndex - blockIndex * RADIX_SORT_BLOCK_SIZE;
			el.localOffset = to - blockHistogram[bucketIndex];
			el.bucket = bucketIndex;
			elementLocations[to] = el;
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
		if( itemIndex < numberOfInputs )
		{
			ElementLocation el = elementLocations[i];
			u32 srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
			u8 bucketIndex = el.bucket;

			u32 dstIndex = pSum[bucketIndex] + el.localOffset;
			outputKeys[dstIndex] = inputKeys[srcIndex];
		}
	}

	if constexpr( keyPair )
	{
		for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
			if( itemIndex < numberOfInputs )
			{
				ElementLocation el = elementLocations[i];
				u32 srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
				u8 bucketIndex = el.bucket;

				u32 dstIndex = pSum[bucketIndex] + el.localOffset;
				outputValues[dstIndex] = inputValues[srcIndex];
			}
		}
	}

	// A special case handling: all elements have the same digit
	//u32 globalOutput = matchMasks[0][0];
	//if( globalOutput-- /* -1 for the actual offset */ )
	//{
	//	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	//	{
	//		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
	//		if( itemIndex < numberOfInputs )
	//		{
	//			u32 dstIndex = globalOutput + i;
	//			outputKeys[dstIndex] = inputKeys[itemIndex];
	//			if constexpr( keyPair )
	//			{
	//				outputValues[dstIndex] = inputValues[itemIndex];
	//			}
	//		}
	//	}
	//	return;
	//}

	// reorder
	//for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	//{
	//	u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
	//	u32 bucketIndex = elementBuckets[i];

	//	__syncthreads();

	//	int warp = threadIdx.x / 32;
	//	int lane = threadIdx.x % 32;

	//	if( itemIndex < numberOfInputs )
	//	{
	//		atomicOr( &matchMasks[warp][bucketIndex], 1u << lane );
	//	}

	//	__syncthreads();

	//	bool flushMask = false;

	//	if( itemIndex < numberOfInputs )
	//	{
	//		u32 matchMask = matchMasks[warp][bucketIndex];
	//		u32 lowerMask = ( 1u << lane ) - 1;
	//		u32 offset = __popc( matchMask & lowerMask );

	//		flushMask = offset == 0;

	//		for( int w = 0; w < warp; w++ )
	//		{
	//			offset += __popc( matchMasks[w][bucketIndex] );
	//		}

	//		u32 localOffset = counters[bucketIndex] + offset;
	//		u32 to = localOffset + localPrefixSum[bucketIndex];

	//		ElementLocation el;
	//		el.localSrcIndex = i;
	//		el.localOffset = localOffset;
	//		el.bucket = bucketIndex;
	//		elementLocations[to] = el;
	//	}

	//	__syncthreads();

	//	if( itemIndex < numberOfInputs )
	//	{
	//		atomicInc( &counters[bucketIndex], 0xFFFFFFFF );
	//	}
	//	if( flushMask )
	//	{
	//		matchMasks[warp][bucketIndex] = 0;
	//	}
	//}

	//for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	//{
	//	u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
	//	if( itemIndex < numberOfInputs )
	//	{
	//		ElementLocation el = elementLocations[i];
	//		u32 srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
	//		u8 bucketIndex = el.bucket;

	//		u32 dstIndex = pSum[bucketIndex] + el.localOffset;
	//		outputKeys[dstIndex] = inputKeys[srcIndex];
	//	}
	//}
	//if constexpr ( keyPair )
	//{
	//	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	//	{
	//		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
	//		if( itemIndex < numberOfInputs )
	//		{
	//			ElementLocation el = elementLocations[i];
	//			u32 srcIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + el.localSrcIndex;
	//			u8 bucketIndex = el.bucket;

	//			u32 dstIndex = pSum[bucketIndex] + el.localOffset;
	//			outputValues[dstIndex] = inputValues[srcIndex];
	//		}
	//	}
	//}
}
extern "C" __global__ void onesweep_reorderKey64( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, u32 numberOfInputs, u32* gpSumBuffer, volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits,
												  u32 iteration )
{
	onesweep_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}
extern "C" __global__ void onesweep_reorderKeyPair64( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* gpSumBuffer,
													  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	onesweep_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}