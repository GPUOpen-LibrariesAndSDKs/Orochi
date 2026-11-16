@echo off
set "OROCHI_OFFLOAD_ARCH=gfx1100"
set "UNITTEST_FILTER=-*getErrorString*"
call "%~dp0unittest.bat"
