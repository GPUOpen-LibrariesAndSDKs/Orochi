@echo off
REM Run Orochi unit tests for Vega 10 (Windows)
REM Usage: unittest_vega10.bat [Debug|DebugFast|RelWithDebInfo|Release]

REM Every path below is relative to this script's directory.
cd /d "%~dp0"
call _config.bat %1
if errorlevel 1 exit /b 1

rd /s /q cache
"%UNITTEST_BIN%" --gtest_filter=-*getErrorString*:*link_bundledBc_with_bc_loweredName* --gtest_output=xml:../result.xml
