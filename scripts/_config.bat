@echo off
REM Shared configuration resolution for the unittest_*.bat runners.
REM Called, not run directly. Sets CONFIG, SUFFIX and UNITTEST_BIN from %1.
REM Usage: call "%~dp0_config.bat" %1

set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Release

set SUFFIX=
if /I "%CONFIG%"=="Debug"          set SUFFIX=D& goto :resolved
if /I "%CONFIG%"=="DebugFast"      set SUFFIX=D& goto :resolved
if /I "%CONFIG%"=="RelWithDebInfo" goto :resolved
if /I "%CONFIG%"=="Release"        goto :resolved

echo error: unknown configuration '%CONFIG%' 1>&2
echo usage: [Debug^|DebugFast^|RelWithDebInfo^|Release] 1>&2
exit /b 1

:resolved
set UNITTEST_BIN=..\dist\bin\%CONFIG%\UnitTest64%SUFFIX%.exe
exit /b 0
