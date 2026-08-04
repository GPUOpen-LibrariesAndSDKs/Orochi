@echo off
REM Run Orochi unit tests for gfx1100 (Windows)
REM Usage: unittest_gfx1100.bat [Debug|DebugFast|RelWithDebInfo|Release]

REM Every path below is relative to this script's directory.
cd /d "%~dp0"
call _config.bat %1
if errorlevel 1 exit /b 1

rd /s /q cache
cd ..\UnitTest\bitcodes
call generate_bitcodes_gfx1100.bat
cd ..\..\scripts
"%UNITTEST_BIN%" --gtest_filter=-*getErrorString* --gtest_output=xml:../result.xml
