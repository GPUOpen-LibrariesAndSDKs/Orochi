//
// Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

// Command line options shared by the Demos and the UnitTest program.
// Kept separate from Common.h so that the UnitTest does not also pull in
// ERROR_CHECK, which implicitly requires a local 'testErrorFlag'.

#include <Orochi/Orochi.h>
#include <charconv>
#include <cstdio>
#include <optional>
#include <string_view>
#include <system_error>


// API requested through a bare 'hip' / 'cuda' argument.
// Without it both APIs stay enabled and Orochi picks one from the devices found.
inline oroApi getApiType( int argc, char** argv )
{
	for( int i = 1; i < argc; ++i )
	{
		const std::string_view arg{ argv[i] };
		if( arg == "hip" )
			return ORO_API_HIP;
		if( arg == "cuda" )
			return ORO_API_CUDA;
	}
	return ( oroApi )( ORO_API_CUDA | ORO_API_HIP );
}


// Whole-string parse, so trailing garbage such as '3x' is rejected rather than
// silently read as 3.
inline std::optional<int> parseDeviceIndex( std::string_view value )
{
	int index = 0;
	const auto [ptr, ec] = std::from_chars( value.data(), value.data() + value.size(), index );
	if( ec != std::errc{} || ptr != value.data() + value.size() )
		return std::nullopt;
	return index;
}


// Orochi device ordinal requested through '--device <n>' / '--device=<n>'.
// Returns 'absentValue' when the option is not on the command line, and
// std::nullopt after reporting a missing or malformed value: silently falling
// back would run the whole program on the wrong GPU.
inline std::optional<int> getDeviceIndex( int argc, char** argv, int absentValue = 0 )
{
	constexpr std::string_view deviceFlag = "--device";
	constexpr std::string_view deviceFlagWithValue = "--device=";

	for( int i = 1; i < argc; ++i )
	{
		const std::string_view arg{ argv[i] };

		std::string_view value;
		if( arg == deviceFlag )
		{
			if( i + 1 >= argc )
			{
				std::printf( "ERROR: '--device' requires a device index\n" );
				return std::nullopt;
			}
			value = argv[i + 1];
		}
		else if( arg.starts_with( deviceFlagWithValue ) )
		{
			value = arg.substr( deviceFlagWithValue.size() );
		}
		else
		{
			continue;
		}

		const std::optional<int> index = parseDeviceIndex( value );
		if( !index )
			std::printf( "ERROR: '%.*s' is not a valid device index\n", ( int )value.size(), value.data() );
		return index;
	}
	return absentValue;
}


// Reports whether 'deviceIndex' names a device visible under the currently
// initialized API set. Must be called after oroInit().
inline bool checkDeviceIndex( int deviceIndex )
{
	int deviceCount = 0;
	if( oroGetDeviceCount( &deviceCount ) != oroSuccess )
	{
		std::printf( "ERROR: unable to query the device count\n" );
		return false;
	}
	if( deviceIndex < 0 || deviceIndex >= deviceCount )
	{
		std::printf( "ERROR: device %d requested but only %d device(s) available\n", deviceIndex, deviceCount );
		return false;
	}
	return true;
}


// Parses '--device', validates it and retrieves the device, so that every Demo
// selects a device the same way. Must be called after oroInit().
inline bool acquireDevice( int argc, char** argv, oroDevice& device )
{
	const std::optional<int> deviceIndex = getDeviceIndex( argc, argv );
	if( !deviceIndex || !checkDeviceIndex( *deviceIndex ) )
		return false;

	if( oroDeviceGet( &device, *deviceIndex ) != oroSuccess )
	{
		std::printf( "ERROR: unable to get device %d\n", *deviceIndex );
		return false;
	}

	std::printf( ">> using device %d\n", *deviceIndex );
	return true;
}
