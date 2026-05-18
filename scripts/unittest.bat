@echo off
REM Run Orochi unit tests (Windows)
REM Cleans cache and runs the test suite excluding known-failing tests.
REM Usage: unittest.bat [Release|Debug|RelWithDebInfo]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release
set SUFFIX=
if "%CONFIG%"=="Debug" set SUFFIX=D

rd /s /q cache
..\dist\bin\Release\Unittest64.exe --gtest_filter=-*link_bundledBc*:*VulkanComputeSimple64* --gtest_output=xml:../result.xml
