@echo off
REM Run Orochi unit tests for Navi 2 (Windows)

rd /s /q cache
..\dist\bin\Release\Unittest64.exe --gtest_filter=-*getErrorString*:*link_bundledBc_with_bc_loweredName* --gtest_output=xml:../result.xml
