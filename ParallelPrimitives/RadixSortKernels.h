#include <ParallelPrimitives/RadixSortConfigs.h>
#define LDS_BARRIER __syncthreads()

using namespace Oro;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// #define NV_WORKAROUND 1

#define THE_FIRST_THREAD threadIdx.x == 0 && blockIdx.x == 0

extern "C" __global__ void CountKernelReference( int* gSrc, int* gDst, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	const int offset = blockIdx.x * blockDim.x * gNItemsPerWI;

	int table[BIN_SIZE] = { 0 };

	for( int i = 0; i < gNItemsPerWI; i++ )
	{
		int idx = offset + threadIdx.x * gNItemsPerWI + i;

		if( idx >= gN ) continue;
		int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
		table[tableIdx]++;
	}

	const int wgIdx = blockIdx.x;

	for( int i = 0; i < BIN_SIZE; i++ )
	{
		if( table[i] != 0 )
		{
			atomicAdd( &gDst[i * N_WGS_EXECUTED + wgIdx], table[i] );
		}
	}
}

//=====

extern "C" __global__ void CountKernel( int* gSrc, int* gDst, int gN, int gNItemsPerWG, const int START_BIT, const int N_WGS_EXECUTED )
{
	__shared__ int table[BIN_SIZE];

	for( int i = threadIdx.x; i < BIN_SIZE; i += COUNT_WG_SIZE )
	{
		table[i] = 0;
	}

	LDS_BARRIER;

	const int offset = blockIdx.x * gNItemsPerWG;
	const int upperBound = ( offset + gNItemsPerWG > gN ) ? gN - offset : gNItemsPerWG;

	for( int i = threadIdx.x; i < upperBound; i += COUNT_WG_SIZE )
	{
		const int idx = offset + i;
		const int tableIdx = ( gSrc[idx] >> START_BIT ) & RADIX_MASK;
		atomicAdd( &table[tableIdx], 1 );
	}

	LDS_BARRIER;

	// Assume COUNT_WG_SIZE == BIN_SIZE
	gDst[threadIdx.x * N_WGS_EXECUTED + blockIdx.x] = table[threadIdx.x];
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
	LDS_BARRIER;

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

		LDS_BARRIER;
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
__device__ void SortImpl( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	int offset = blockIdx.x * blockDim.x * gNItemsPerWI;
	if( offset > gN )
	{
		return;
	}

	__shared__ u32 localOffsets[BIN_SIZE];
	__shared__ u32 ldsKeys[SORT_WG_SIZE * SORT_N_ITEMS_PER_WI];
	__shared__ u32 ldsValues[KEY_VALUE_PAIR ? SORT_WG_SIZE * SORT_N_ITEMS_PER_WI : 1];

	__shared__ union
	{
		u16 histogram[2][BIN_SIZE]; // low and high// todo. can be aliased
		u32 histogramU32[BIN_SIZE];
	} lds;

	int keys[SORT_N_ITEMS_PER_WI] = { 0 };
	int values[KEY_VALUE_PAIR ? SORT_N_ITEMS_PER_WI : 1] = { 0 };

	for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
	{
		localOffsets[i] = gHistogram[i * N_WGS_EXECUTED + blockIdx.x];
	}
	LDS_BARRIER;

	for( int ii = 0; ii < gNItemsPerWI; ii += SORT_N_ITEMS_PER_WI )
	{
		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int idx = offset + i * SORT_WG_SIZE + threadIdx.x;
			ldsKeys[i * SORT_WG_SIZE + threadIdx.x] = ( idx < gN ) ? gSrcKey[idx] : 0xffffffff;

			if constexpr( KEY_VALUE_PAIR )
			{
				ldsValues[i * SORT_WG_SIZE + threadIdx.x] = ( idx < gN ) ? gSrcVal[idx] : 0xffffffff;
			}
		}
		LDS_BARRIER;

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int idx = threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			keys[i] = ldsKeys[idx];

			if constexpr( KEY_VALUE_PAIR )
			{
				values[i] = ldsValues[idx];
			}
		}

		localSort8bitMulti<KEY_VALUE_PAIR>( keys, ldsKeys, values, ldsValues, START_BIT );

		for( int i = threadIdx.x; i < BIN_SIZE; i += SORT_WG_SIZE )
		{
			lds.histogramU32[i] = 0;
		}
		LDS_BARRIER;

		for( int i = 0; i < SORT_N_ITEMS_PER_WI; ++i )
		{
			const int a = threadIdx.x * SORT_N_ITEMS_PER_WI + i;
			const int b = a - 1;
			const int aa = ( ldsKeys[a] >> START_BIT ) & RADIX_MASK;
			const int bb = ( ( ( b >= 0 ) ? ldsKeys[b] : 0xffffffff ) >> START_BIT ) & RADIX_MASK;
			if( aa != bb )
			{
				lds.histogram[0][aa] = a;
				if( b >= 0 ) lds.histogram[1][bb] = a;
			}
		}
		if( threadIdx.x == 0 ) lds.histogram[1][( ldsKeys[SORT_N_ITEMS_PER_WI * SORT_WG_SIZE - 1] >> START_BIT ) & RADIX_MASK] = SORT_N_ITEMS_PER_WI * SORT_WG_SIZE;

		LDS_BARRIER;

		const int upperBound = ( offset + threadIdx.x * SORT_N_ITEMS_PER_WI + SORT_N_ITEMS_PER_WI > gN ) ? ( gN - ( offset + threadIdx.x * SORT_N_ITEMS_PER_WI ) ) : SORT_N_ITEMS_PER_WI;
		if( upperBound < 0 )
		{
			return;
		}

		for( int i = 0; i < upperBound; ++i )
		{
			const int tableIdx = ( keys[i] >> START_BIT ) & RADIX_MASK;
			const int dstIdx = localOffsets[tableIdx] + ( threadIdx.x * SORT_N_ITEMS_PER_WI + i ) - lds.histogram[0][tableIdx];
			gDstKey[dstIdx] = keys[i];

			if constexpr( KEY_VALUE_PAIR )
			{
				gDstVal[dstIdx] = values[i];
			}
		}

		LDS_BARRIER;

		for( int i = 0; i < N_BINS_PER_WI; i++ )
		{
			const int idx = threadIdx.x * N_BINS_PER_WI + i;
			localOffsets[idx] += lds.histogram[1][idx] - lds.histogram[0][idx];
		}

		offset += SORT_WG_SIZE * SORT_N_ITEMS_PER_WI;
		if( offset > gN )
		{
			return;
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

extern "C" __global__ void SortSinglePassKernel( int* gSrcKey, int* gDstKey, int gN, const int START_BIT, const int END_BIT ) 
{
	SortSinglePass<false>( gSrcKey, nullptr, gDstKey, nullptr, gN, START_BIT, END_BIT ); 
}

extern "C" __global__ void SortSinglePassKVKernel( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int gN, const int START_BIT, const int END_BIT ) 
{ 
	SortSinglePass<true>( gSrcKey, gSrcVal, gDstKey, gDstVal, gN, START_BIT, END_BIT ); 
}

extern "C" __global__ void SortKernel( int* gSrcKey, int* gDstKey, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	SortImpl<false>( gSrcKey, nullptr, gDstKey, nullptr, gHistogram, gN, gNItemsPerWI, START_BIT, N_WGS_EXECUTED );
}

extern "C" __global__ void SortKVKernel( int* gSrcKey, int* gSrcVal, int* gDstKey, int* gDstVal, int* gHistogram, int gN, int gNItemsPerWI, const int START_BIT, const int N_WGS_EXECUTED )
{
	SortImpl<true>( gSrcKey, gSrcVal, gDstKey, gDstVal, gHistogram, gN, gNItemsPerWI, START_BIT, N_WGS_EXECUTED );
}

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

	LDS_BARRIER;
}

extern "C" __global__ void ParallelExclusiveScanAllWG( int* gCount, int* gHistogram, volatile int* gPartialSum, volatile bool* gIsReady )
{
	// Fill the LDS with the partial sum of each segment
	__shared__ int blockBuffer[SCAN_WG_SIZE];

	blockBuffer[threadIdx.x] = gCount[blockIdx.x * blockDim.x + threadIdx.x];

	LDS_BARRIER;

	// Do parallel exclusive scan on the LDS

	int currentSegmentSum = ldsScanExclusive( blockBuffer, SCAN_WG_SIZE );

	LDS_BARRIER;

	// Sync all the Workgroups to calculate the global offset.

	__shared__ int currentGlobalOffset;
	WorkgroupSync( threadIdx.x, blockIdx.x, currentSegmentSum, &currentGlobalOffset, gPartialSum, gIsReady );

	// Write back the result.

	gHistogram[blockIdx.x * blockDim.x + threadIdx.x] = blockBuffer[threadIdx.x] + currentGlobalOffset;
}
