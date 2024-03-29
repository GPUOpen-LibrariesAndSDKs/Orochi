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

#include "demosTest.h"
#include "common.h"
#include <filesystem>

void FormatPathForOS(std::string& path)
{
#ifdef _WIN32
	for(int i=0; i<path.size(); i++)
	{
		if ( path[i] == '/' )
			path[i] = '\\';
	}
#endif
	return;
}

void ExecDemo(const std::string& testName)
{
	std::string programName = "../dist/bin/"
		
	#ifdef _DEBUG
		+ std::string("Debug/")
	#else
		+ std::string("Release/")
	#endif
		
		+ testName

	#ifdef _DEBUG
		+ std::string("D")
	#endif

	#ifdef _WIN32
		+ std::string(".exe")
	#endif
		;

	FormatPathForOS(programName);

	if ( !std::filesystem::exists(programName) )
	{
		std::cout << "Error: The Demo \"" << programName << "\" program file doesn't exist" << std::endl;
		ASSERT_TRUE(0);
	}

	int retCode = std::system(  std::string( "\"" + programName + "\"" ).c_str()  );

	#ifdef _WIN32
	if ( retCode != OROCHI_TEST_RETCODE__SUCCESS )
	#else
	if ( WEXITSTATUS(retCode) != OROCHI_TEST_RETCODE__SUCCESS ) // for Unix, return code needs to be processed to get the original return value.
	#endif
	{
		std::cout << "Error: The Demo \"" << programName << "\" program returned an error return code: " << retCode << std::endl;
		ASSERT_TRUE(0);
	}
	return;
}


TEST_F( OroDemoBase, SimpleDemo64 )
{
	std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ExecDemo(testName);
	return;
}

TEST_F( OroDemoBase, Texture64 )
{
	std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ExecDemo(testName);
	return;
}

TEST_F( OroDemoBase, DeviceEnum64 )
{
	std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ExecDemo(testName);
	return;
}

TEST_F( OroDemoBase, WMMA64 )
{
	std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ExecDemo(testName);
	return;
}


//
// Tests specific to Windows
//
#ifdef _WIN32 


TEST_F( OroDemoBase, VulkanComputeSimple64 )
{
	std::string testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
	ExecDemo(testName);
	return;
}


#endif






