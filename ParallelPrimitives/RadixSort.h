#pragma once

#include <Orochi/GpuMemory.h>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSortConfigs.h>
#include <Test/Stopwatch.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace Oro
{

class RadixSort final
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

	RadixSort( oroDevice device, OrochiUtils& oroutils, oroStream stream = 0, const std::string& kernelPath = "", const std::string& includeDir = "" );

	// Allow move but disallow copy.
	RadixSort( RadixSort&& ) noexcept = default;
	RadixSort& operator=( RadixSort&& ) noexcept = default;
	RadixSort( const RadixSort& ) = delete;
	RadixSort& operator=( const RadixSort& ) = delete;
	~RadixSort() = default;

	void setFlag( Flag flag ) noexcept;

	void sort( KeyValueSoA src, KeyValueSoA dst, uint32_t n, int startBit, int endBit, oroStream stream = 0 ) noexcept;

	void sort( u32* src, u32* dst, uint32_t n, int startBit, int endBit, oroStream stream = 0 ) noexcept;

  private:
	//template<class T>
	//void sort1pass( const T src, const T dst, int n, int startBit, int endBit, oroStream stream ) noexcept;

	///// @brief Compile the kernels for radix sort.
	///// @param kernelPath The kernel path.
	///// @param includeDir The include directory.
	void compileKernels( const std::string& kernelPath, const std::string& includeDir ) noexcept;

	//[[nodiscard]] int calculateWGsToExecute( const int blockSize ) const noexcept;

	///// @brief Exclusive scan algorithm on CPU for testing.
	///// It copies the count result from the Device to Host before computation, and then copies the offsets back from Host to Device afterward.
	///// @param countsGpu The count result in GPU memory. Otuput: The offset.
	///// @param offsetsGpu The offsets.
	//void exclusiveScanCpu( const Oro::GpuMemory<int>& countsGpu, Oro::GpuMemory<int>& offsetsGpu ) const noexcept;

	/// @brief Configure the settings, compile the kernels and allocate the memory.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	void configure( const std::string& kernelPath, const std::string& includeDir, oroStream stream ) noexcept;

  private:
	//// GPU blocks for the count kernel
	//int m_num_blocks_for_count{};

	//// GPU blocks for the scan kernel
	//int m_num_blocks_for_scan{};

	Flag m_flags{ Flag::NO_LOG };

	enum class Kernel
	{
		//COUNT,
		//SCAN_SINGLE_WG,
		//SCAN_PARALLEL,
		//SORT,
		//SORT_KV,
		SORT_SINGLE_PASS,
		SORT_SINGLE_PASS_KV,
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;

	/// @brief  The enum class which indicates the selected algorithm of prefix scan.
	//enum class ScanAlgo
	//{
	//	SCAN_CPU,
	//	SCAN_GPU_SINGLE_WG,
	//	SCAN_GPU_PARALLEL,
	//};

	//constexpr static auto selectedScanAlgo{ ScanAlgo::SCAN_GPU_PARALLEL };

	//GpuMemory<int> m_partial_sum;
	//GpuMemory<bool> m_is_ready;

	oroDevice m_device{};
	//oroDeviceProp m_props{};

	OrochiUtils& m_oroutils;

	// This buffer holds the "bucket" table from all GPU blocks.
	//GpuMemory<int> m_tmp_buffer;

	//int m_num_threads_per_block_for_count{};
	//int m_num_threads_per_block_for_scan{};
	//int m_num_threads_per_block_for_sort{};

	//int m_num_warps_per_block_for_sort{};

	oroFunction m_gHistogram;
	oroFunction m_gPrefixSum;
	oroFunction m_onesweep_reorderKey64;
	oroFunction m_onesweep_reorderKeyPair64;

	GpuMemory<uint8_t> m_tmpBuffer;
};
}; // namespace Oro
