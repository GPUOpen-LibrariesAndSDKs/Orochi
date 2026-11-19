#pragma once
#include <Orochi/Orochi.h>
#include <Test/CommandLine.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// return true if error
inline bool checkError( oroError e )
{
	if( e != oroSuccess )
	{
		const char* pStr = nullptr;
		oroGetErrorString( e, &pStr );
		std::printf("ERROR==================\n");
		if ( pStr )
			std::printf("%s\n", pStr);
		else
			std::printf("<No Error String>\n");
		return true;
	}
	return false;
}

// return true if error
inline bool checkError( orortcResult e )
{
	if ( e != ORORTC_SUCCESS )
	{
		std::printf("ERROR in RTC==================\n");
		return true;
	}
	return false;
}

#define ERROR_CHECK( e ) if( checkError(e) ) testErrorFlag=true;

