@echo off
REM Run Orochi unit tests for Navi 2 (Windows)
REM Usage: unittest_navi2.bat [Release|Debug|RelWithDebInfo]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release
set SUFFIX=
if "%CONFIG%"=="Debug" set SUFFIX=D

rd /s /q cache
..\dist\bin\%CONFIG%\UnitTest64%SUFFIX%.exe --gtest_filter=-*getErrorString*:*link_bundledBc_with_bc_loweredName* --gtest_output=xml:../result.xml
