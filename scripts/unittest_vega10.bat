@echo off
REM Run Orochi unit tests for Vega 10 (Windows)
REM Usage: unittest_vega10.bat [Release|Debug|RelWithDebInfo]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release
set SUFFIX=
if "%CONFIG%"=="Debug" set SUFFIX=D

rd /s /q cache
..\dist\bin\%CONFIG%\Unittest64%SUFFIX%.exe --gtest_filter=-*getErrorString*:*link_bundledBc_with_bc_loweredName* --gtest_output=xml:../result.xml
