#pragma once
#include <Orochi/Orochi.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

inline
oroApi getApiType( int argc, char** argv )
{
	// By default, the 2 API are enabled, and will be automatically selected by Orochi depending on the devices.
	oroApi api = ( oroApi )( ORO_API_CUDA | ORO_API_HIP ); 

	if( argc >= 2 )
	{
		if( strcmp( argv[1], "hip" ) == 0 )
			api = ORO_API_HIP;
		if( strcmp( argv[1], "cuda" ) == 0 )
			api = ORO_API_CUDA;
	}
	return api;
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

