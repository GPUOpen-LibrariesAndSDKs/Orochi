@echo off
REM Run Orochi unit tests for gfx1102 (Windows)
REM Usage: unittest_gfx1102.bat [Release|Debug|RelWithDebInfo]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release
set SUFFIX=
if "%CONFIG%"=="Debug" set SUFFIX=D

rd /s /q cache
cd ..\UnitTest\bitcodes
call generate_bitcodes_gfx1102.bat
cd ..\..\scripts
..\dist\bin\%CONFIG%\UnitTest64%SUFFIX% --gtest_filter=-*getErrorString* --gtest_output=xml:../result.xml
