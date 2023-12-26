#pragma once

namespace Oro
{
constexpr auto WG_SIZE{ 64 };
constexpr auto N_RADIX{ 8 };
constexpr auto BIN_SIZE{ 1 << N_RADIX };
constexpr auto RADIX_MASK{ ( 1 << N_RADIX ) - 1 };
constexpr auto PACK_FACTOR{ sizeof( int ) / sizeof( char ) };
constexpr auto N_PACKED{ BIN_SIZE / PACK_FACTOR };
constexpr auto PACK_MAX{ 255 };
constexpr auto N_PACKED_PER_WI{ N_PACKED / WG_SIZE };
constexpr auto N_BINS_PER_WI{ BIN_SIZE / WG_SIZE };
constexpr auto N_BINS_4BIT{ 16 };
constexpr auto N_BINS_PACK_FACTOR{ sizeof( long long ) / sizeof( short ) };
constexpr auto N_BINS_PACKED_4BIT{ N_BINS_4BIT / N_BINS_PACK_FACTOR };

constexpr auto N_BINS_8BIT{ 1 << 8 };

constexpr auto DEFAULT_WARP_SIZE{ 32 };

constexpr auto DEFAULT_NUM_WARPS_PER_BLOCK{ 8 };

// count config

constexpr auto DEFAULT_COUNT_BLOCK_SIZE{ DEFAULT_WARP_SIZE * DEFAULT_NUM_WARPS_PER_BLOCK };

// scan configs
constexpr auto DEFAULT_SCAN_BLOCK_SIZE{ DEFAULT_WARP_SIZE * DEFAULT_NUM_WARPS_PER_BLOCK };

// sort configs
constexpr auto DEFAULT_SORT_BLOCK_SIZE{ DEFAULT_WARP_SIZE * DEFAULT_NUM_WARPS_PER_BLOCK };
constexpr auto SORT_N_ITEMS_PER_WI{ 12 };
constexpr auto SINGLE_SORT_N_ITEMS_PER_WI{ 24 };
constexpr auto SINGLE_SORT_WG_SIZE{ 128 };

// Checks

static_assert( BIN_SIZE % 2 == 0 );

// Notice that, on some GPUs, the max size of a GPU block cannot be greater than 256
static_assert( DEFAULT_COUNT_BLOCK_SIZE % DEFAULT_WARP_SIZE == 0 );
static_assert( DEFAULT_SCAN_BLOCK_SIZE % DEFAULT_WARP_SIZE == 0 );

constexpr int RADIX_SORT_BLOCK_SIZE = 2048;

constexpr int GHISTOGRAM_ITEM_PER_BLOCK = 2048;
constexpr int GHISTOGRAM_THREADS_PER_BLOCK = 256;

constexpr int REORDER_NUMBER_OF_WARPS = 8;
constexpr int REORDER_NUMBER_OF_THREADS_PER_BLOCK = 32 * REORDER_NUMBER_OF_WARPS;

constexpr int LOOKBACK_TABLE_SIZE = 1024;
constexpr int MAX_LOOK_BACK = 64;
constexpr int TAIL_BITS = 4;
constexpr int TAIL_COUNT = 1u << TAIL_BITS;

}; // namespace Oro