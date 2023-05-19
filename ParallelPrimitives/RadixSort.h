#pragma once

#include <Orochi/GpuMemory.h>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSortConfigs.h>
#include <Test/Stopwatch.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

// #define PROFILE 1

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

	enum class Flag
	{
		NO_LOG,
		LOG,
	};

	RadixSort();

	// Allow move but disallow copy.
	RadixSort( RadixSort&& ) noexcept = default;
	RadixSort& operator=( RadixSort&& ) noexcept = default;
	RadixSort( const RadixSort& ) = delete;
	RadixSort& operator=( const RadixSort& ) = delete;
	~RadixSort();

	/// @brief Configure the settings, compile the kernels and allocate the memory.
	/// @param device The device.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	/// @return The size of the temp buffer.
	u32 configure( oroDevice device, OrochiUtils& oroutils, const std::string& kernelPath = "", const std::string& includeDir = "", oroStream stream = 0 ) noexcept;

	void setFlag( Flag flag ) noexcept;

	void sort( const KeyValueSoA src, const KeyValueSoA dst, int n, int startBit, int endBit, u32* tempBuffer, oroStream stream = 0 ) noexcept;

	void sort( const u32* src, const u32* dst, int n, int startBit, int endBit, u32* tempBuffer, oroStream stream = 0 ) noexcept;

  private:
	template<class T>
	void sort1pass( const T src, const T dst, int n, int startBit, int endBit, int* temps, oroStream stream ) noexcept;

	/// @brief Compile the kernels for radix sort.
	/// @param device The device.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	void compileKernels( oroDevice device, OrochiUtils& oroutils, const std::string& kernelPath, const std::string& includeDir ) noexcept;

	int calculateWGsToExecute( oroDevice device ) noexcept;

	/// @brief Exclusive scan algorithm on CPU for testing.
	/// It copies the count result from the Device to Host before computation, and then copies the offsets back from Host to Device afterward.
	/// @param countsGpu The count result in GPU memory. Otuput: The offset.
	/// @param offsetsGpu The offsets.
	/// @param nWGsToExecute Number of WGs to execute
	void exclusiveScanCpu( int* countsGpu, int* offsetsGpu, const int nWGsToExecute, oroStream stream ) noexcept;

  private:
	int m_nWGsToExecute{ 4 };
	Flag m_flags{ Flag::NO_LOG };

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

	GpuMemory<int> m_partialSum;
	bool* m_isReady{ nullptr };
};

#include <ParallelPrimitives/RadixSort.inl>

}; // namespace Oro
