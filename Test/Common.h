#pragma once
#include <Orochi/Orochi.h>
#include <string.h>

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