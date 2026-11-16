@echo off
set "UNITTEST_FILTER=-*getErrorString*:*link_bundledBc_with_bc_loweredName*"
call "%~dp0unittest.bat"
