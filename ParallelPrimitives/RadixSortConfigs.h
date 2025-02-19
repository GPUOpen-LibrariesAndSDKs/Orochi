#pragma once

namespace Oro
{
constexpr auto WG_SIZE{ 64 };
constexpr auto N_RADIX{ 8 };
constexpr auto BIN_SIZE{ 1 << N_RADIX };
constexpr auto RADIX_MASK{ ( 1 << N_RADIX ) - 1 };
constexpr auto PACK_FACTOR{ sizeof( int ) / sizeof( char ) };
constexpr auto N_BINS_4BIT{ 16 };
constexpr auto N_BINS_PACK_FACTOR{ sizeof( long long ) / sizeof( short ) };
constexpr auto N_BINS_PACKED_4BIT{ N_BINS_4BIT / N_BINS_PACK_FACTOR };

// sort configs
constexpr auto SORT_N_ITEMS_PER_WI{ 12 };
constexpr auto SINGLE_SORT_N_ITEMS_PER_WI{ 24 };
constexpr auto SINGLE_SORT_WG_SIZE{ 128 };

// Checks

static_assert( BIN_SIZE % 2 == 0 );

constexpr int WARP_SIZE = 32;

constexpr int RADIX_SORT_BLOCK_SIZE = 4096;

constexpr int GHISTOGRAM_ITEM_PER_BLOCK = 2048;
constexpr int GHISTOGRAM_THREADS_PER_BLOCK = 256;
constexpr int GHISTOGRAM_ITEMS_PER_THREAD = GHISTOGRAM_ITEM_PER_BLOCK / GHISTOGRAM_THREADS_PER_BLOCK;

constexpr int REORDER_NUMBER_OF_WARPS = 8;
constexpr int REORDER_NUMBER_OF_THREADS_PER_BLOCK = WARP_SIZE * REORDER_NUMBER_OF_WARPS;
constexpr int REORDER_NUMBER_OF_ITEM_PER_WARP = RADIX_SORT_BLOCK_SIZE / REORDER_NUMBER_OF_WARPS;
constexpr int REORDER_NUMBER_OF_ITEM_PER_THREAD = REORDER_NUMBER_OF_ITEM_PER_WARP / 32;

constexpr int LOOKBACK_TABLE_SIZE = 1024;
constexpr int MAX_LOOK_BACK = 64;
constexpr int TAIL_BITS = 5;
constexpr auto TAIL_MASK = 0xFFFFFFFFu << TAIL_BITS;
static_assert( MAX_LOOK_BACK < LOOKBACK_TABLE_SIZE, "" );

//static_assert( BIN_SIZE <= REORDER_NUMBER_OF_THREADS_PER_BLOCK, "please check scanExclusive" );
//static_assert( BIN_SIZE % REORDER_NUMBER_OF_THREADS_PER_BLOCK == 0, "please check prefixSumExclusive on onesweep_reorder" );

}; // namespace Oro