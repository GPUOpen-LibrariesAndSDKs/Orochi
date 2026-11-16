@echo off
rem Runs UnitTest64; the tests open sources by "../UnitTest/..." relative to scripts/.
rem UNITTEST_FILTER overrides the gtest filter, OROCHI_OFFLOAD_ARCH the bitcode arch list.
setlocal
cd /d "%~dp0"

if exist cache rd /s /q cache

rem A generator exits 127 when its toolchain is absent; any other failure aborts.
set GENERATED=0
for %%G in (generate_bitcodes.bat generate_bitcodes_nvidia.bat) do (
	call ..\UnitTest\bitcodes\%%G
	if errorlevel 128 (
		echo error: %%G failed 1>&2
		exit /b 1
	)
	if errorlevel 127 (
		rem toolchain absent
	) else if errorlevel 1 (
		echo error: %%G failed 1>&2
		exit /b 1
	) else (
		set GENERATED=1
	)
)
if "%GENERATED%"=="0" (
	echo error: bitcodes were not generated: neither a ROCm nor a CUDA toolchain was found 1>&2
	exit /b 1
)

if not defined UNITTEST_FILTER set "UNITTEST_FILTER=-*link_bundledBc*:*VulkanComputeSimple64*"
..\dist\bin\Release\UnitTest64.exe --gtest_filter=%UNITTEST_FILTER% --gtest_output=xml:../result.xml
