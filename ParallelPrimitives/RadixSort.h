#pragma once

#include <Orochi/Orochi.h>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <ParallelPrimitives/RadixSortConfigs.h>

//#define PROFILE 1

namespace Oro
{

class RadixSort
{
  public:
	using u32 = uint32_t;
	using u64 = uint64_t;

	struct KeyValueSoA
	{
		u32* key;
		u32* value;
	};

	enum Flag
	{
		FLAG_LOG = 1 << 0,
	};

	RadixSort();

	// Allow move but disallow copy.
	RadixSort( RadixSort&& ) = default;
	RadixSort& operator=( RadixSort&& ) = default;
	RadixSort( const RadixSort& ) = delete;
	RadixSort& operator=( const RadixSort& ) = delete;
	~RadixSort();

	void configure( oroDevice device, u32& tempBufferSizeOut );

	void setFlag( Flag flag );

	void sort( const KeyValueSoA src, const KeyValueSoA dst, int n, int startBit, int endBit, u32* tempBuffer ) noexcept;

	void sort( const u32* src, const u32* dst, int n, int startBit, int endBit, u32* tempBuffer ) noexcept;

  private:
	template<class T>
	void sort1pass( const T src, const T dst, int n, int startBit, int endBit, int* temps ) noexcept;

	void compileKernels( oroDevice device );

	int calculateWGsToExecute( oroDevice device ) noexcept;

  private:
	int m_nWGsToExecute{ 4 };
	Flag m_flags;

	enum class Kernel
	{
		COUNT,
		COUNT_REF,
		SCAN_SINGLE_WG,
		SCAN_PARALLEL,
		SORT,
		SORT_KV,
		SORT_SINGLE_PASS,
		SORT_SINGLE_PASS_KV,
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;

	/// @brief  The enum class which indicates the selected algorithm of prefix scan.
	enum class ScanAlgo
	{
		SCAN_CPU,
		SCAN_GPU_SINGLE_WG,
		SCAN_GPU_PARALLEL,
	};

	constexpr static auto selectedScanAlgo{ ScanAlgo::SCAN_GPU_PARALLEL };

	int* m_partialSum{ nullptr };
	bool* m_isReady{ nullptr };
};

#include <ParallelPrimitives/RadixSort.inl>

}; // namespace Oro
