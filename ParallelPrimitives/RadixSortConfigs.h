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

constexpr auto DEFAULT_NUM_WARP_PER_BLOCK{ 8 };

// count config

constexpr auto DEFAULT_COUNT_BLOCK_SIZE{ DEFAULT_WARP_SIZE * DEFAULT_NUM_WARP_PER_BLOCK };

// scan configs
constexpr auto DEFAULT_SCAN_BLOCK_SIZE{ DEFAULT_WARP_SIZE * DEFAULT_NUM_WARP_PER_BLOCK };

// sort configs
constexpr auto SORT_WG_SIZE{ 64 };
constexpr auto SORT_N_ITEMS_PER_WI{ 12 };
constexpr auto SINGLE_SORT_N_ITEMS_PER_WI{ 24 };
constexpr auto SINGLE_SORT_WG_SIZE{ 128 };

// Checks

// Notice that, on some GPUs, the max size of a GPU block cannot be greater than 256
static_assert( DEFAULT_COUNT_BLOCK_SIZE % DEFAULT_WARP_SIZE == 0 );
static_assert( DEFAULT_SCAN_BLOCK_SIZE % DEFAULT_WARP_SIZE == 0 );

}; // namespace Oro