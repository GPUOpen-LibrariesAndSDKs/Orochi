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
	oroApi api = ORO_API_HIP;
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
	const char* pStr = nullptr;
	oroGetErrorString( e, &pStr );
	if( e != oroSuccess )
	{
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

