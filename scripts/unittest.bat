@echo off
REM Run Orochi unit tests (Windows)
REM Cleans cache and runs the test suite excluding known-failing tests.
REM Usage: unittest.bat [Debug|DebugFast|RelWithDebInfo|Release]

REM Every path below is relative to this script's directory.
cd /d "%~dp0"
call _config.bat %1
if errorlevel 1 exit /b 1

rd /s /q cache
"%UNITTEST_BIN%" --gtest_filter=-*link_bundledBc*:*VulkanComputeSimple64* --gtest_output=xml:../result.xml
