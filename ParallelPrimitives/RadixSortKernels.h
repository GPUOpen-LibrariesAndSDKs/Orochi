#include <ParallelPrimitives/RadixSortConfigs.h>
#define LDS_BARRIER __syncthreads()

namespace
{

using namespace Oro;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
} // namespace

// #define NV_WORKAROUND 1

__device__ constexpr u32 getMaskedBits( const u32 value, const u32 shift ) noexcept { return ( value >> shift ) & RADIX_MASK; }

extern "C" __global__ void CountKernel( int* gSrc, int* gDst, int gN, int gNItemsPerWG, const int START_BIT, const int N_WGS_EXECUTED )
{
	__shared__ int table[BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += COUNT_WG_SIZE )
	{
		table[i] = 0;
	}

	__syncthreads();

	const int offset = blockIdx.x * gNItemsPerWG;
	const int upperBound = ( offset + gNItemsPerWG > gN ) ? gN - offset : gNItemsPerWG;

	for( int i = threadIdx.x; i < upperBound; i += COUNT_WG_SIZE )
	{
		const int idx = offset + i;
		const int tableIdx = getMaskedBits( gSrc[idx], START_BIT );
		atomicAdd( &table[tableIdx], 1 );
	}

	__syncthreads();

	for( int i = threadIdx.x; i < BIN_SIZE; i += COUNT_WG_SIZE )
	{
		gDst[i * N_WGS_EXECUTED + blockIdx.x] = table[i];
	}
}

template<typename T, int STRIDE>
struct ScanImpl
{
	__device__ static T exec( T a )
	{
		T b = __shfl( a, threadIdx.x - STRIDE );
		if( threadIdx.x >= STRIDE ) a += b;
		return ScanImpl<T, STRIDE * 2>::exec( a );
	}
};

template<typename T>
struct ScanImpl<T, WG_SIZE>
{
	__device__ static T exec( T a ) { return a; }
};

template<typename T>
__device__ void waveScanInclusive( T& a, int width )
{
#if 0
	a = ScanImpl<T, 1>::exec( a );
#else
	for( int i = 1; i < width; i *= 2 )
	{
		T b = __shfl( a, threadIdx.x - i );
		if( threadIdx.x >= i ) a += b;
	}
#endif
}

template<typename T>
__device__ T waveScanExclusive( T& a, int width )
{
	waveScanInclusive( a, width );

	T sum = __shfl( a, width - 1 );
	a = __shfl( a, threadIdx.x - 1 );
	if( threadIdx.x == 0 ) a = 0;

	return sum;
}

template<typename T>
__device__ void ldsScanInclusive( T* lds, int width )
{
	// The width cannot exceed WG_SIZE
	__shared__ T temp[2][WG_SIZE];

	constexpr int MAX_INDEX = 1;
	int outIndex = 0;
	int inIndex = 1;

	temp[outIndex][threadIdx.x] = lds[threadIdx.x];
	__syncthreads();

	for( int i = 1; i < width; i *= 2 )
	{
		// Swap in and out index for the buffers

		outIndex = MAX_INDEX - outIndex;
		inIndex = MAX_INDEX - outIndex;

		if( threadIdx.x >= i )
		{
			temp[outIndex][threadIdx.x] = temp[inIndex][threadIdx.x] + temp[inIndex][threadIdx.x - i];
		}
		else
		{
			temp[outIndex][threadIdx.x] = temp[inIndex][threadIdx.x];
		}

		__syncthreads();
	}

	lds[threadIdx.x] = temp[outIndex][threadIdx.x];

	// Ensure the results are written in LDS and are observable in a block (workgroup) before return.
	__threadfence_block();
}

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

__device__ void localSort8bitMulti_shared_bin( int* keys, u32* ldsKeys, const int START_BIT )
{
	__shared__ unsigned table[BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
	{
		table[i] = 0U;
	}

	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
	{
		const int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
		atomicAdd( &table[tableIdx], 1 );
	}

	LDS_BARRIER;

	int globalSum = 0;
	for( int binId = 0; binId < BIN_SIZE; binId += SORT_WG_SIZE * 2 )
	{
		unsigned* globalOffset = &table[binId];
		const unsigned currentGlobalSum = ldsScanExclusive( globalOffset, SORT_WG_SIZE * 2 );
		globalOffset[threadIdx.x * 2] += globalSum;
		globalOffset[threadIdx.x * 2 + 1] += globalSum;
		globalSum += currentGlobalSum;
	}

	LDS_BARRIER;

	__shared__ u32 keyBuffer[SORT_WG_SIZE * SORT_N_ITEMS_PER_WI];

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
	{
		keyBuffer[threadIdx.x * SORT_N_ITEMS_PER_WI + i] = keys[i];
	}

	LDS_BARRIER;

	if( threadIdx.x == 0 )
	{
		for( int i = 0; i < SORT_WG_SIZE * SORT_N_ITEMS_PER_WI; ++i )
		{
			const int tableIdx = ( keyBuffer[i] >> START_BIT ) & RADIX_MASK;
			const int writeIndex = table[tableIdx];

			ldsKeys[writeIndex] = keyBuffer[i];

			++table[tableIdx];
		}
	}

	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

__device__ void localSort8bitMulti_group( int* keys, u32* ldsKeys, const int START_BIT )
{
	constexpr auto N_GROUP_SIZE{ N_BINS_8BIT / ( sizeof( u64 ) / sizeof( u16 ) ) };

	__shared__ union
	{
		u16 m_ungrouped[SORT_WG_SIZE + 1][N_BINS_8BIT];
		u64 m_grouped[SORT_WG_SIZE + 1][N_GROUP_SIZE];
	} lds;

	for( int i = 0; i < N_GROUP_SIZE; ++i )
	{
		lds.m_grouped[threadIdx.x][i] = 0U;
	}

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		const auto in8bit = ( keys[i] >> START_BIT ) & RADIX_MASK;
		++lds.m_ungrouped[threadIdx.x][in8bit];
	}

	LDS_BARRIER;

	for( int groupId = threadIdx.x; groupId < N_GROUP_SIZE; groupId += SORT_WG_SIZE )
	{
		u64 sum = 0U;
		for( int i = 0; i < SORT_WG_SIZE; i++ )
		{
			const auto current = lds.m_grouped[i][groupId];
			lds.m_grouped[i][groupId] = sum;
			sum += current;
		}
		lds.m_grouped[SORT_WG_SIZE][groupId] = sum;
	}

	LDS_BARRIER;

	int globalSum = 0;
	for( int binId = 0; binId < N_BINS_8BIT; binId += SORT_WG_SIZE * 2 )
	{
		auto* globalOffset = &lds.m_ungrouped[SORT_WG_SIZE][binId];
		const int currentGlobalSum = ldsScanExclusive( globalOffset, SORT_WG_SIZE * 2 );
		globalOffset[threadIdx.x * 2] += globalSum;
		globalOffset[threadIdx.x * 2 + 1] += globalSum;
		globalSum += currentGlobalSum;
	}

	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		const auto in8bit = ( keys[i] >> START_BIT ) & RADIX_MASK;
		const auto offset = lds.m_ungrouped[SORT_WG_SIZE][in8bit];
		const auto rank = lds.m_ungrouped[threadIdx.x][in8bit]++;

		ldsKeys[offset + rank] = keys[i];
	}

	LDS_BARRIER;

	for( int i = 0; i < SORT_N_ITEMS_PER_WI; i++ )
	{
		keys[i] = ldsKeys[threadIdx.x * SORT_N_ITEMS_PER_WI + i];
	}
}

template<bool KEY_VALUE_PAIR>
__device__ void localSort8bitMulti( int* keys, u32* ldsKeys, int* values, u32* ldsValues, const int START_BIT )
{
	localSort4bitMulti<SORT_N_ITEMS_PER_WI, SORT_WG_SIZE, KEY_VALUE_PAIR>( keys, ldsKeys, values, ldsValues, START_BIT );
	if( N_RADIX > 4 ) localSort4bitMulti<SORT_N_ITEMS_PER_WI, SORT_WG_SIZE, KEY_VALUE_PAIR>( keys, ldsKeys, values, ldsValues, START_BIT + 4 );
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

extern "C" __global__ void ParallelExclusiveScanSingleWG( int* gCount, int* gHistogram, const int N_WGS_EXECUTED )
{
	// Use a single WG.
	if( blockIdx.x != 0 )
	{
		return;
	}

	// LDS for the parallel scan of the global sum:
	// First we store the sum of the counters of each number to it,
	// then we compute the global offset using parallel exclusive scan.
	__shared__ int blockBuffer[BIN_SIZE];

	// fill the LDS with the local sum

	for( int binId = threadIdx.x; binId < BIN_SIZE; binId += WG_SIZE )
	{
		// Do exclusive scan for each segment handled by each WI in a WG

		int localThreadSum = 0;
		for( int i = 0; i < N_WGS_EXECUTED; ++i )
		{
			int current = gCount[binId * N_WGS_EXECUTED + i];
			gCount[binId * N_WGS_EXECUTED + i] = localThreadSum;

			localThreadSum += current;
		}

		// Store the thread local sum to LDS.

		blockBuffer[binId] = localThreadSum;
	}

	LDS_BARRIER;

	// Do parallel exclusive scan on the LDS

	int globalSum = 0;
	for( int binId = 0; binId < BIN_SIZE; binId += WG_SIZE * 2 )
	{
		int* globalOffset = &blockBuffer[binId];
		int currentGlobalSum = ldsScanExclusive( globalOffset, WG_SIZE * 2 );
		globalOffset[threadIdx.x * 2] += globalSum;
		globalOffset[threadIdx.x * 2 + 1] += globalSum;
		globalSum += currentGlobalSum;
	}

	LDS_BARRIER;

	// Add the global offset to the global histogram.

	for( int binId = threadIdx.x; binId < BIN_SIZE; binId += WG_SIZE )
	{
		for( int i = 0; i < N_WGS_EXECUTED; ++i )
		{
			gHistogram[binId * N_WGS_EXECUTED + i] += blockBuffer[binId];
		}
	}
}

extern "C" __device__ void WorkgroupSync( int threadId, int blockId, int currentSegmentSum, int* currentGlobalOffset, volatile int* gPartialSum, volatile bool* gIsReady )
{
	if( threadId == 0 )
	{
		int offset = 0;

		if( blockId != 0 )
		{
			while( !gIsReady[blockId - 1] )
			{
			}

			offset = gPartialSum[blockId - 1];

			__threadfence();

			// Reset the value
			gIsReady[blockId - 1] = false;
		}

		gPartialSum[blockId] = offset + currentSegmentSum;

		// Ensure that the gIsReady is only modified after the gPartialSum is written.
		__threadfence();

		gIsReady[blockId] = true;

		*currentGlobalOffset = offset;
	}

	__syncthreads();
}

extern "C" __global__ void ParallelExclusiveScanAllWG( int* gCount, int* gHistogram, volatile int* gPartialSum, volatile bool* gIsReady )
{
	// Fill the LDS with the partial sum of each segment
	__shared__ int blockBuffer[SCAN_WG_SIZE];

	blockBuffer[threadIdx.x] = gCount[blockIdx.x * blockDim.x + threadIdx.x];

	__syncthreads();

	// Do parallel exclusive scan on the LDS

	int currentSegmentSum = ldsScanExclusive( blockBuffer, SCAN_WG_SIZE );

	__syncthreads();

	// Sync all the Workgroups to calculate the global offset.

	__shared__ int currentGlobalOffset;
	WorkgroupSync( threadIdx.x, blockIdx.x, currentSegmentSum, &currentGlobalOffset, gPartialSum, gIsReady );

	// Write back the result.

	gHistogram[blockIdx.x * blockDim.x + threadIdx.x] = blockBuffer[threadIdx.x] + currentGlobalOffset;
}

constexpr auto REORDER_NUMBER_OF_WARPS = DEFAULT_NUM_WARP_PER_BLOCK;

// TODO: Note that the parameter is changed from gNItemsPerWI to gNItemsPerWG
// Need to change the launch function accordingly

template<bool KEY_VALUE_PAIR>
__device__ void SortImpl( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int* gHistogram, int numberOfInputs, int gNItemsPerWG, const int START_BIT, const int N_WGS_EXECUTED )
{
	__shared__ u32 globalOffset[BIN_SIZE];
	__shared__ u32 localPrefixSum[BIN_SIZE];
	__shared__ u32 counters[BIN_SIZE];

	__shared__ u32 matchMasks[REORDER_NUMBER_OF_WARPS][BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
	{
		// Note: The size of gHistogram is always BIN_SIZE * N_WGS_EXECUTED
		globalOffset[i] = gHistogram[i * N_WGS_EXECUTED + blockIdx.x];

		counters[i] = 0;
		localPrefixSum[i] = 0;
	}

	for( int w = 0; w < REORDER_NUMBER_OF_WARPS; ++w )
	{
		for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
		{
			matchMasks[w][i] = 0;
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < gNItemsPerWG; i += SORT_WG_SIZE )
	{
		const u32 itemIndex = blockIdx.x * gNItemsPerWG + i;
		if( itemIndex < numberOfInputs )
		{
			const auto item = gSrcKey[itemIndex];
			const u32 bucketIndex = getMaskedBits( item, START_BIT );
			atomicInc( &localPrefixSum[bucketIndex], 0xFFFFFFFF );
		}
	}

	__syncthreads();

	// Compute Prefix Sum

	ldsScanExclusive( localPrefixSum, BIN_SIZE );

	__syncthreads();

	// Reorder

	for( int i = threadIdx.x; i < gNItemsPerWG; i += SORT_WG_SIZE )
	{
		const u32 itemIndex = blockIdx.x * gNItemsPerWG + i;

		// TODO:
		const auto item = gSrcKey[itemIndex];
		const u32 bucketIndex = getMaskedBits( item, START_BIT );

		const int warp = threadIdx.x / 32;
		const int lane = threadIdx.x % 32;

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicOr( &matchMasks[warp][bucketIndex], 1u << lane );
		}

		__syncthreads();

		bool flushMask = false;

		u32 localOffset = 0;
		u32 localSrcIndex = 0;

		if( itemIndex < numberOfInputs )
		{
			const u32 matchMask = matchMasks[warp][bucketIndex];
			const u32 lowerMask = ( 1u << lane ) - 1;
			u32 offset = __popc( matchMask & lowerMask );

			flushMask = ( offset == 0 );

			for( int w = 0; w < warp; ++w )
			{
				offset += __popc( matchMasks[w][bucketIndex] );
			}

			localOffset = counters[bucketIndex] + offset;
			localSrcIndex = i;
		}

		__syncthreads();

		if( itemIndex < numberOfInputs )
		{
			atomicInc( &counters[bucketIndex], 0xFFFFFFFF );
		}

		if( flushMask )
		{
			matchMasks[warp][bucketIndex] = 0;
		}

		// Swap

		// TODO:
		if( itemIndex < numberOfInputs )
		{
			const u32 srcIndex = blockIdx.x * gNItemsPerWG + localSrcIndex;
			const u32 dstIndex = globalOffset[bucketIndex] + localOffset;
			gDstKey[dstIndex] = gSrcKey[srcIndex];

			if constexpr( KEY_VALUE_PAIR )
			{
				gDstVal[dstIndex] = gSrcVal[srcIndex];
			}
		}
	}
}

extern "C" __global__ void SortKernel( int* gSrcKey, int* gDstKey, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	SortImpl<false>( gSrcKey, nullptr, gDstKey, nullptr, gHistogram, gN, gNItemsPerWI, START_BIT, N_WGS_EXECUTED );
}

extern "C" __global__ void SortKVKernel( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	SortImpl<true>( gSrcKey, gSrcVal, gDstKey, gDstVal, gHistogram, gN, gNItemsPerWI, START_BIT, N_WGS_EXECUTED );
}
