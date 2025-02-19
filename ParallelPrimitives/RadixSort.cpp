#include <ParallelPrimitives/RadixSort.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>

// if ORO_PP_LOAD_FROM_STRING &&     ORO_PRECOMPILED -> we load the precompiled/baked kernels.
// if ORO_PP_LOAD_FROM_STRING && NOT ORO_PRECOMPILED -> we load the baked source code kernels (from Kernels.h / KernelArgs.h)
#if !defined( ORO_PRECOMPILED ) && defined( ORO_PP_LOAD_FROM_STRING )
// Note: the include order must be in this particular form.
// clang-format off
#include <ParallelPrimitives/cache/Kernels.h>
#include <ParallelPrimitives/cache/KernelArgs.h>
// clang-format on
#else
// if Kernels.h / KernelArgs.h are not included, declare nullptr strings
static const char* hip_RadixSortKernels = nullptr;
namespace hip
{
static const char** RadixSortKernelsArgs = nullptr;
static const char** RadixSortKernelsIncludes = nullptr;
} // namespace hip
#endif

#if defined( __GNUC__ )
#include <dlfcn.h>
#endif

#if defined( ORO_PRECOMPILED ) && defined( ORO_PP_LOAD_FROM_STRING )
#include <ParallelPrimitives/cache/oro_compiled_kernels.h> // generate this header with 'convert_binary_to_array.py'
#else
const unsigned char oro_compiled_kernels_h[] = "";
const size_t oro_compiled_kernels_h_size = 0;
#endif

constexpr uint64_t div_round_up64( uint64_t val, uint64_t divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }
constexpr uint64_t next_multiple64( uint64_t val, uint64_t divisor ) noexcept { return div_round_up64( val, divisor ) * divisor; }

namespace
{

// if those 2 preprocessors are enabled, this activates the 'usePrecompiledAndBakedKernel' mode.
#if defined( ORO_PRECOMPILED ) && defined( ORO_PP_LOAD_FROM_STRING )

// this flag means that we bake the precompiled kernels
constexpr auto usePrecompiledAndBakedKernel = true;

constexpr auto useBitCode = false;
constexpr auto useBakeKernel = false;

#else

constexpr auto usePrecompiledAndBakedKernel = false;

#if defined( ORO_PRECOMPILED )
constexpr auto useBitCode = true;	 // this flag means we use the bitcode file
#else
constexpr auto useBitCode = false;
#endif

#if defined( ORO_PP_LOAD_FROM_STRING )
constexpr auto useBakeKernel = true; // this flag means we use the HIP source code embeded in the binary ( as a string )
#else
constexpr auto useBakeKernel = false;
#endif

#endif

static_assert( !( useBitCode && useBakeKernel ), "useBitCode and useBakeKernel cannot coexist" );

#if !defined( __GNUC__ )
const HMODULE GetCurrentModule()
{
	HMODULE hModule = NULL;
	// hModule is NULL if GetModuleHandleEx fails.
	GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCTSTR)GetCurrentModule, &hModule );
	return hModule;
}
#else
void GetCurrentModule1() {}
#endif

void printKernelInfo( const std::string& name, oroFunction func )
{
	std::cout << "Function: " << name;

	int numReg{};
	int sharedSizeBytes{};
	int constSizeBytes{};
	oroFuncGetAttribute( &numReg, ORO_FUNC_ATTRIBUTE_NUM_REGS, func );
	oroFuncGetAttribute( &sharedSizeBytes, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func );
	oroFuncGetAttribute( &constSizeBytes, ORO_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func );
	std::cout << ", vgpr : shared = " << numReg << " : " << sharedSizeBytes << " : " << constSizeBytes << '\n';
}

} // namespace

namespace Oro
{

RadixSort::RadixSort( oroDevice device, OrochiUtils& oroutils, oroStream stream, const std::string& kernelPath, const std::string& includeDir ) : m_device{ device }, m_oroutils{ oroutils }
{
	oroGetDeviceProperties( &m_props, device );
	configure( kernelPath, includeDir, stream );
}

void RadixSort::compileKernels( const std::string& kernelPath, const std::string& includeDir ) noexcept
{
	static constexpr auto defaultKernelPath{ "../ParallelPrimitives/RadixSortKernels.h" };
	static constexpr auto defaultIncludeDir{ "../" };

	const auto currentKernelPath{ ( kernelPath == "" ) ? defaultKernelPath : kernelPath };
	const auto currentIncludeDir{ ( includeDir == "" ) ? defaultIncludeDir : includeDir };

	const auto getCurrentDir = []() noexcept
	{
#if !defined( __GNUC__ )
		HMODULE hm = GetCurrentModule();
		char buff[MAX_PATH];
		GetModuleFileName( hm, buff, MAX_PATH );
#else
		Dl_info info;
		dladdr( (const void*)GetCurrentModule1, &info );
		const char* buff = info.dli_fname;
#endif
		std::string::size_type position = std::string( buff ).find_last_of( "\\/" );
		return std::string( buff ).substr( 0, position ) + "/";
	};

	std::string binaryPath{};
	std::string log{};
	if constexpr( useBitCode )
	{
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		binaryPath = getCurrentDir();
		binaryPath += isAmd ? "oro_compiled_kernels.hipfb" : "oro_compiled_kernels.fatbin";
		log = "loading pre-compiled kernels at path : " + binaryPath;
	}
	else
	{
		log = "compiling kernels at path : " + currentKernelPath + " in : " + currentIncludeDir;
	}

	if( m_flags == Flag::LOG )
	{
		std::cout << log << std::endl;
	}

	const auto includeArg{ "-I" + currentIncludeDir };
	std::vector<const char*> opts;
	opts.push_back( includeArg.c_str() );

	struct Record
	{
		std::string kernelName;
		Kernel kernelType;
	};

	const std::vector<Record> records{ { "SortSinglePassKernel", Kernel::SORT_SINGLE_PASS },
									   { "SortSinglePassKVKernel", Kernel::SORT_SINGLE_PASS_KV },
									   { "GHistogram", Kernel::SORT_GHISTOGRAM },
									   { "OnesweepReorderKey64", Kernel::SORT_ONESWEEP_REORDER_KEY_64 },
									   { "OnesweepReorderKeyPair64", Kernel::SORT_ONESWEEP_REORDER_KEY_PAIR_64 } };


	for( const auto& record : records )
	{
		if constexpr( usePrecompiledAndBakedKernel ) 
		{ 
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromPrecompiledBinary_asData( oro_compiled_kernels_h, oro_compiled_kernels_h_size, record.kernelName.c_str() ); 
		}
		else if constexpr( useBakeKernel )
		{
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromString( m_device, hip_RadixSortKernels, currentKernelPath.c_str(), record.kernelName.c_str(), &opts, 1, hip::RadixSortKernelsArgs, hip::RadixSortKernelsIncludes );
		}
		else if constexpr( useBitCode )
		{
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromPrecompiledBinary( binaryPath.c_str(), record.kernelName.c_str() );
		}
		else
		{
			oroFunctions[record.kernelType] = m_oroutils.getFunctionFromFile( m_device, currentKernelPath.c_str(), record.kernelName.c_str(), &opts );
		}

		if( m_flags == Flag::LOG )
		{
			printKernelInfo( record.kernelName, oroFunctions[record.kernelType] );
		}
	}

}

void RadixSort::configure( const std::string& kernelPath, const std::string& includeDir, oroStream stream ) noexcept
{
	compileKernels( kernelPath, includeDir );

	u64 gpSumBuffer = sizeof( u32 ) * BIN_SIZE * sizeof( u32 /* key type */ );
	m_gpSumBuffer.resizeAsync( gpSumBuffer, false /*copy*/, stream );

	u64 lookBackBuffer = sizeof( u64 ) * ( BIN_SIZE * LOOKBACK_TABLE_SIZE );
	m_lookbackBuffer.resizeAsync( lookBackBuffer, false /*copy*/, stream );

	m_tailIterator.resizeAsync( 1, false /*copy*/, stream );
	m_tailIterator.resetAsync( stream );
	m_gpSumCounter.resizeAsync( 1, false /*copy*/, stream );
}
void RadixSort::setFlag( Flag flag ) noexcept { m_flags = flag; }

void RadixSort::sort( const KeyValueSoA& src, const KeyValueSoA& dst, uint32_t n, int startBit, int endBit, oroStream stream ) noexcept
{
	bool keyPair = src.value != nullptr;

	// todo. better to compute SINGLE_SORT_N_ITEMS_PER_WI which we use in the kernel dynamically rather than hard coding it to distribute the work evenly
	// right now, setting this as large as possible is faster than multi pass sorting
	if( n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI )
	{
		if( keyPair )
		{
			const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS_KV];
			const void* args[] = { &src.key, &src.value, &dst.key, &dst.value, &n, &startBit, &endBit };
			OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		}
		else
		{
			const auto func = oroFunctions[Kernel::SORT_SINGLE_PASS];
			const void* args[] = { &src, &dst, &n, &startBit, &endBit };
			OrochiUtils::launch1D( func, SINGLE_SORT_WG_SIZE, args, SINGLE_SORT_WG_SIZE, 0, stream );
		}
		return;
	}

	int nIteration = div_round_up64( endBit - startBit, 8 );
	uint64_t numberOfBlocks = div_round_up64( n, RADIX_SORT_BLOCK_SIZE );

	m_lookbackBuffer.resetAsync( stream );
	m_gpSumCounter.resetAsync( stream );
	m_gpSumBuffer.resetAsync( stream );

	// counter for gHistogram. 
	{
		int maxBlocksPerMP = 0;
		oroError e = oroModuleOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerMP, oroFunctions[Kernel::SORT_GHISTOGRAM], GHISTOGRAM_THREADS_PER_BLOCK, 0 );
		const int nBlocks = e == oroSuccess ? maxBlocksPerMP * m_props.multiProcessorCount : 2048;

		const void* args[] = { &src.key, &n, arg_cast( m_gpSumBuffer.address() ), &startBit, arg_cast( m_gpSumCounter.address() ) };
		OrochiUtils::launch1D( oroFunctions[Kernel::SORT_GHISTOGRAM], nBlocks * GHISTOGRAM_THREADS_PER_BLOCK, args, GHISTOGRAM_THREADS_PER_BLOCK, 0, stream );
	}

	auto s = src;
	auto d = dst;
	for( int i = 0; i < nIteration; i++ )
	{
		if( numberOfBlocks < LOOKBACK_TABLE_SIZE * 2 )
		{
			m_lookbackBuffer.resetAsync( stream );
		} // other wise, we can skip zero clear look back buffer

		if( keyPair )
		{
			const void* args[] = { &s.key, &d.key, &s.value, &d.value, &n, arg_cast( m_gpSumBuffer.address() ), arg_cast( m_lookbackBuffer.address() ), arg_cast( m_tailIterator.address() ), &startBit, &i };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT_ONESWEEP_REORDER_KEY_PAIR_64], numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		else
		{
			const void* args[] = { &s.key, &d.key, &n, arg_cast( m_gpSumBuffer.address() ), arg_cast( m_lookbackBuffer.address() ), arg_cast( m_tailIterator.address() ), &startBit, &i };
			OrochiUtils::launch1D( oroFunctions[Kernel::SORT_ONESWEEP_REORDER_KEY_64], numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		std::swap( s, d );
	}

	if( s.key == src.key )
	{
		oroMemcpyDtoDAsync( (oroDeviceptr)dst.key, (oroDeviceptr)src.key, sizeof( uint32_t ) * n, stream );

		if( keyPair )
		{
			oroMemcpyDtoDAsync( (oroDeviceptr)dst.value, (oroDeviceptr)src.value, sizeof( uint32_t ) * n, stream );
		}
	}
}

void RadixSort::sort( u32* src, u32* dst, uint32_t n, int startBit, int endBit, oroStream stream ) noexcept
{
	sort( KeyValueSoA{ src, nullptr }, KeyValueSoA{ dst, nullptr }, n, startBit, endBit, stream );
}
}; // namespace Oro
