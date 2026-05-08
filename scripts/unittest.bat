@echo off
REM Run Orochi unit tests (Windows)
REM Cleans cache and runs the test suite excluding known-failing tests.

rd /s /q cache
..\dist\bin\Release\Unittest64.exe --gtest_filter=-*link_bundledBc*:*VulkanComputeSimple64*:*checkCUEW* --gtest_output=xml:../result.xml
