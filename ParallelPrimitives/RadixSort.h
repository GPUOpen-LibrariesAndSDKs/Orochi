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

	void sort( const KeyValueSoA& src, const KeyValueSoA& dst, uint32_t n, int startBit, int endBit, oroStream stream = 0 ) noexcept;

	void sort( u32* src, u32* dst, uint32_t n, int startBit, int endBit, oroStream stream = 0 ) noexcept;

  private:
	// @brief Compile the kernels for radix sort.
	// @param kernelPath The kernel path.
	// @param includeDir The include directory.
	void compileKernels( const std::string& kernelPath, const std::string& includeDir ) noexcept;

	/// @brief Configure the settings, compile the kernels and allocate the memory.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	void configure( const std::string& kernelPath, const std::string& includeDir, oroStream stream ) noexcept;

  private:
	Flag m_flags{ Flag::NO_LOG };

	enum class Kernel
	{
		SORT_SINGLE_PASS,
		SORT_SINGLE_PASS_KV,
	};

	std::unordered_map<Kernel, oroFunction> oroFunctions;

	oroDevice m_device{};
	oroDeviceProp m_props{};

	OrochiUtils& m_oroutils;

	oroFunction m_gHistogram;
	oroFunction m_gPrefixSum;
	oroFunction m_onesweep_reorderKey64;
	oroFunction m_onesweep_reorderKeyPair64;

	GpuMemory<uint8_t> m_lookbackBuffer;
	GpuMemory<uint8_t> m_gpSumBuffer;
	GpuMemory<u32> m_gpSumCounter;
	GpuMemory<u32> m_tailIterator;
};
}; // namespace Oro
