//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <Orochi/Orochi.h>
#include <Test/Common.h>

#include <Orochi/OrochiUtils.h>
#include <ParallelPrimitives/RadixSort.h>
#include <ParallelPrimitives/RadixSortConfigs.h>
#include <algorithm>
#include <vector>
#include <numeric>
#if 1
#include <Test/Stopwatch.h>
#else
#include <chrono>

class Stopwatch
{
  public:
	void start() { m_start = std::chrono::system_clock::now(); }
	void stop() { m_end = std::chrono::system_clock::now(); }
	float getMs() { return std::chrono::duration_cast<std::chrono::milliseconds>( m_end - m_start ).count(); }

  private:
	std::chrono::time_point<std::chrono::system_clock> m_start, m_end;
};
#endif

using u64 = Oro::RadixSort::u64;
using u32 = Oro::RadixSort::u32;

class SortTest
{
  public:
	SortTest( oroDevice dev, oroCtx ctx, OrochiUtils& oroutils ) : m_device( dev ), m_ctx( ctx ), m_sort(dev, oroutils)
	{
	}

	~SortTest() { OrochiUtils::free( m_tempBuffer ); }

	template<bool KEY_VALUE_PAIR = true>
	void test( int testSize, const int testBits = 32, const int nRuns = 1 )
	{
		srand( 123 );
		Oro::RadixSort::KeyValueSoA srcGpu{};
		Oro::RadixSort::KeyValueSoA dstGpu{};

		OrochiUtils::malloc( srcGpu.key, testSize );
		OrochiUtils::malloc( dstGpu.key, testSize );
		void* temp;
		oroMalloc( (oroDeviceptr*)&temp, m_sort.getRequiredTemporalStorageBytes( testSize ) );

		std::vector<u32> srcKey( testSize );
		for( int i = 0; i < testSize; i++ )
		{
			srcKey[i] = getRandom( 0u, (u32)( ( 1ull << (u64)testBits ) - 1 ) );
		}

		std::vector<u32> srcValue( testSize );
		if constexpr( KEY_VALUE_PAIR )
		{
			OrochiUtils::malloc( srcGpu.value, testSize );
			OrochiUtils::malloc( dstGpu.value, testSize );

			for( int i = 0; i < testSize; i++ )
			{
				srcValue[i] = getRandom( 0u, (u32)( ( 1ull << (u64)testBits ) - 1 ) );
			}
		}

		Stopwatch sw;
		for( int i = 0; i < nRuns; i++ )
		{
			OrochiUtils::copyHtoD( srcGpu.key, srcKey.data(), testSize );
			OrochiUtils::waitForCompletion();

			if constexpr( KEY_VALUE_PAIR )
			{
				OrochiUtils::copyHtoD( srcGpu.value, srcValue.data(), testSize );
				OrochiUtils::waitForCompletion();
			}

			sw.start();

			if constexpr( KEY_VALUE_PAIR )
			{
				m_sort.sort( srcGpu, dstGpu, testSize, 0, testBits, temp );
			}
			else
			{
				m_sort.sort( srcGpu.key, dstGpu.key, testSize, 0, testBits, temp );
			}

			OrochiUtils::waitForCompletion();
			sw.stop();
			float ms = sw.getMs();
			float gKeys_s = static_cast<float>(testSize) / 1000.f / 1000.f / ms;
			printf( "%5.2fms (%3.2fGKeys/s) sorting %3.1fMkeys [%s]\n", ms, gKeys_s, testSize / 1000.f / 1000.f, KEY_VALUE_PAIR? "keyValue":"key" );
		}

		std::vector<u32> dstKey( testSize );
		OrochiUtils::copyDtoH( dstKey.data(), dstGpu.key, testSize );

		std::vector<u32> dstValue( testSize );
		if constexpr( KEY_VALUE_PAIR )
		{
			OrochiUtils::copyDtoH( dstValue.data(), dstGpu.value, testSize );
		}

		std::vector<u32> indexHelper( testSize );
		std::iota( std::begin( indexHelper ), std::end( indexHelper ), 0U );

		std::stable_sort( std::begin( indexHelper ), std::end( indexHelper ), [&]( const auto indexA, const auto indexB ) noexcept { return srcKey[indexA] < srcKey[indexB]; } );

		const auto rearrange = []( auto& targetBuffer, const auto& indexBuffer ) noexcept
		{
			std::vector<u32> tmpBuffer( std::size( targetBuffer ) );

			for( auto i = 0UL; i < std::size( targetBuffer ); ++i )
			{
				tmpBuffer[i] = targetBuffer[indexBuffer[i]];
			}

			targetBuffer = std::move( tmpBuffer );
		};

		rearrange( srcKey, indexHelper );
		if constexpr( KEY_VALUE_PAIR )
		{
			rearrange( srcValue, indexHelper );
		}

		const auto check = [&]( const size_t i ) noexcept
		{
			if constexpr( KEY_VALUE_PAIR )
			{
				return dstKey[i] != srcKey[i] || dstValue[i] != srcValue[i];
			}
			else
			{
				return dstKey[i] != srcKey[i];
			}
		};

		for( int i = 0; i < testSize; i++ )
		{
			if( check( i ) )
			{
				printf( "fail at %d\n", i );
				__debugbreak();
				break;
			}
		}

		if constexpr( KEY_VALUE_PAIR )
		{
			OrochiUtils::free( srcGpu.value );
			OrochiUtils::free( dstGpu.value );
		}

		OrochiUtils::free( srcGpu.key );
		OrochiUtils::free( dstGpu.key );
		oroFree( (oroDeviceptr)temp );

		printf( "passed: %3.2fK keys\n", testSize / 1000.f );
	}

	template<typename T>
	inline T getRandom( const T minV, const T maxV )
	{
		double r = std::min( (double)RAND_MAX - 1, (double)rand() ) / RAND_MAX;
		T range = maxV - minV;
		return (T)( minV + r * range );
	}

  private:
	oroDevice m_device;
	oroCtx m_ctx;
	Oro::RadixSort m_sort;
	u32* m_tempBuffer;
};

enum TestType
{
	TEST_SMALL, // test single kernel sort
	TEST_SIMPLE,
	TEST_PERF,
	TEST_BITS,
};

int main( int argc, char** argv )
{
	TestType testType = TEST_PERF;
	oroApi api = getApiType( argc, argv );

	int a = oroInitialize( api, 0 );
	if( a != 0 )
	{
		printf( "initialization failed\n" );
		return 0;
	}
	printf( ">> executing on %s\n", ( api == ORO_API_HIP ) ? "hip" : "cuda" );

	printf( ">> testing initialization\n" );
	oroError e;
	e = oroInit( 0 );
	oroDevice device;
	e = oroDeviceGet( &device, 0 );
	oroCtx ctx;
	e = oroCtxCreate( &ctx, 0, device );

	printf( ">> testing device props\n" );
	{
		oroDeviceProp props;
		oroGetDeviceProperties( &props, device );
		int v;
		oroDriverGetVersion( &v );
		printf( "executing on %s (%s), %d SIMDs (driverVer.:%d)\n", props.name, props.gcnArchName, props.multiProcessorCount, v );
	}

	OrochiUtils oroutils;
	SortTest sort( device, ctx, oroutils );
	const int testBits = 32;
	switch( testType )
	{
	case TEST_SMALL:
	{
		for( int i = 0; i < 10; i++ )
			sort.test<false>( 64 * 10 * ( i + 1 ), testBits, 2 );
		for( int i = 0; i < 10; i++ )
			sort.test<true>( 64 * 10 * ( i + 1 ), testBits, 2 );
	}
	break;
	case TEST_SIMPLE:
		sort.test( 16 * 1000 * 100, testBits );
		break;
	case TEST_PERF:
	{
		const int nRuns = 4;
		sort.test( 16 * 1000 * 10, testBits, nRuns );
		sort.test( 16 * 1000 * 100, testBits, nRuns );
		sort.test( 16 * 1000 * 1000, testBits, nRuns );

		sort.test<false>( 16 * 1000 * 10, testBits, nRuns );
		sort.test<false>( 16 * 1000 * 100, testBits, nRuns );
		sort.test<false>( 16 * 1000 * 1000, testBits, nRuns );
		printf( ">> testing 16 bit sort\n" );
		const int testBits = 16;
		sort.test( 16 * 1000 * 10, testBits, nRuns );
		sort.test( 16 * 1000 * 100, testBits, nRuns );
		sort.test( 16 * 1000 * 1000, testBits, nRuns );

		sort.test<false>( 16 * 1000 * 10, testBits, nRuns );
		sort.test<false>( 16 * 1000 * 100, testBits, nRuns );
		sort.test<false>( 16 * 1000 * 1000, testBits, nRuns );
	}
	break;
	case TEST_BITS:
	{
		const int nRuns = 2;
		int testSize = 16 * 1000 * 1000;
		sort.test( testSize, 8, nRuns );
		sort.test( testSize, 16, nRuns );
		sort.test( testSize, 24, nRuns );
		sort.test( testSize, 32, nRuns );
	}
	break;
	};

	printf( ">> done\n" );
	return 0;
}
