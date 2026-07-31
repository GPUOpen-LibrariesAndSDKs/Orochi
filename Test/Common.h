#pragma once
#include <Orochi/Orochi.h>
#include <charconv>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

inline
oroApi getApiType( int argc, char** argv )
{
	// By default, the 2 API are enabled, and will be automatically selected by Orochi depending on the devices.
	oroApi api = ( oroApi )( ORO_API_CUDA | ORO_API_HIP ); 

	for( int i = 1; i < argc; ++i )
	{
		const std::string_view arg{ argv[i] };
		if( arg == "hip" )
			api = ORO_API_HIP;
		if( arg == "cuda" )
			api = ORO_API_CUDA;
	}
	return api;
}

// Orochi device ordinal requested through '--device <n>' / '--device=<n>'.
// Returns 'defaultIndex' when the option is absent or its value is not a number.
inline
int getDeviceIndex( int argc, char** argv, int defaultIndex = 0 )
{
	constexpr std::string_view kFlag = "--device";
	constexpr std::string_view kFlagWithValue = "--device=";

	const auto parse = []( std::string_view value, int fallback )
	{
		int index = 0;
		const auto [ptr, ec] = std::from_chars( value.data(), value.data() + value.size(), index );
		const bool parsed = ec == std::errc{} && ptr == value.data() + value.size();
		return parsed ? index : fallback;
	};

	for( int i = 1; i < argc; ++i )
	{
		const std::string_view arg{ argv[i] };
		if( arg == kFlag && i + 1 < argc )
			return parse( argv[i + 1], defaultIndex );
		if( arg.starts_with( kFlagWithValue ) )
			return parse( arg.substr( kFlagWithValue.size() ), defaultIndex );
	}
	return defaultIndex;
}

// Reports whether 'deviceIndex' names a device visible under the currently initialized API set.
// Must be called after oroInit().
inline
bool checkDeviceIndex( int deviceIndex )
{
	int deviceCount = 0;
	if( oroGetDeviceCount( &deviceCount ) != oroSuccess )
	{
		printf( "ERROR: unable to query the device count\n" );
		return false;
	}
	if( deviceIndex < 0 || deviceIndex >= deviceCount )
	{
		printf( "ERROR: device %d requested but only %d device(s) available\n", deviceIndex, deviceCount );
		return false;
	}
	return true;
}

// return true if error
inline bool checkError( oroError e )
{
	if( e != oroSuccess )
	{
		const char* pStr = nullptr;
		oroGetErrorString( e, &pStr );
		printf("ERROR==================\n");
		if ( pStr )
			printf("%s\n", pStr);
		else
			printf("<No Error String>\n");
		return true;
	}
	return false;
}

// return true if error
inline bool checkError( orortcResult e )
{
	if ( e != ORORTC_SUCCESS )
	{
		printf("ERROR in RTC==================\n");
		return true;
	}
	return false;
}

#define ERROR_CHECK( e ) if( checkError(e) ) testErrorFlag=true;

