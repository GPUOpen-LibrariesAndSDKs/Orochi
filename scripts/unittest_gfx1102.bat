@echo off
set "OROCHI_OFFLOAD_ARCH=gfx1102"
set "UNITTEST_FILTER=-*getErrorString*"
call "%~dp0unittest.bat"
