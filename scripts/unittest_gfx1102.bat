@echo off
REM Run Orochi unit tests for gfx1102 (Windows)

rd /s /q cache
cd ..\UnitTest\bitcodes
call generate_bitcodes_gfx1102.bat
cd ..\..\scripts
..\dist\bin\Release\Unittest64 --gtest_filter=-*getErrorString* --gtest_output=xml:../result.xml
