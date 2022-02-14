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

inline void checkError( oroError e )
{
	const char* pStr;
	oroGetErrorString( e, &pStr );
	if( e != oroSuccess ) 
		printf( "ERROR==================\n%s\n", pStr);
}

#define ERROR_CHECK( e ) checkError( e )