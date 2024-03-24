//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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


#include <gtest/gtest.h>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <Orochi/GpuMemory.h>
#include <fstream>
#include "demoErrorCodes.h"

#if defined( OROASSERT )
	#undef OROASSERT
#endif
#define OROASSERT( x ) ASSERT_TRUE( x )
#define OROCHECK( x ) { oroError e = x; ASSERT_EQ( e , ORO_SUCCESS ); }
#define ORORTCCHECK( x ) { ASSERT_EQ( x , ORORTC_SUCCESS ); }


// Base class used by most of the unit test.
// This class manages the usual initialization/destructions 
class OroTestBase : public ::testing::Test
{
  public:
	void SetUp() 
	{
		const int deviceIndex = 0;
		oroApi api = ( oroApi )( ORO_API_CUDA | ORO_API_HIP );
		int a = oroInitialize( api, 0 );
		OROASSERT( a == 0 );

		OROCHECK( oroInit( 0 ) );
		OROCHECK( oroDeviceGet( &m_device, deviceIndex ) );
		OROCHECK( oroCtxCreate( &m_ctx, 0, m_device ) );
		OROCHECK( oroCtxSetCurrent( m_ctx ) );
		OROCHECK( oroStreamCreate( &m_stream ) );

		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		m_jitLogVerbose = isAmd ? 1 : 0; // on CUDA, if using '1', orortcLinkComplete crashes... (driver 546.33 / CUDA 12.2)

	}

	void TearDown() 
	{ 
		OROCHECK( oroStreamDestroy( m_stream ) );
		OROCHECK( oroCtxDestroy( m_ctx ) );
	}

  protected:
	oroDevice m_device = 0;
	oroCtx m_ctx = nullptr;
	oroStream m_stream = nullptr;

	int32_t m_jitLogVerbose = 1; // used for ORORTC_JIT_LOG_VERBOSE
};


// Use this base class for unit test that takes care to create all the Orochi environement by themselves.
// Usually used to play Demos 
class OroDemoBase : public ::testing::Test
{
  public:
	void SetUp() 
	{
	}

	void TearDown() 
	{ 
	}

};


