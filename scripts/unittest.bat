@echo off
rem Runs UnitTest64; the tests open sources by "../UnitTest/..." relative to scripts/.
rem UNITTEST_FILTER overrides the gtest filter, OROCHI_OFFLOAD_ARCH the bitcode arch list.
setlocal
cd /d "%~dp0"

if exist cache rd /s /q cache

rem A generator whose toolchain is absent is skipped; any other failure only warns.
set GENERATED=0

if defined ROCM_PATH (set "ROCM=%ROCM_PATH%") else (set "ROCM=%HIP_PATH%")
set "HAVE_ROCM="
for %%P in ("%ROCM%\lib\llvm\bin\amdclang++.exe" "%ROCM%\bin\amdclang++.exe" "%ROCM%\bin\hipcc.exe") do if exist %%P set HAVE_ROCM=1
if defined HAVE_ROCM (
	call :gen generate_bitcodes.bat
) else (
	echo note: skipping generate_bitcodes.bat: ROCm not found 1>&2
)

set "NVCC=nvcc"
if defined CUDA_PATH set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
set "HAVE_NVCC="
where "%NVCC%" >nul 2>&1 && set HAVE_NVCC=1
if exist "%NVCC%" set HAVE_NVCC=1
if defined HAVE_NVCC (
	call :gen generate_bitcodes_nvidia.bat
) else (
	echo note: skipping generate_bitcodes_nvidia.bat: nvcc not found 1>&2
)

if "%GENERATED%"=="0" (
	echo error: bitcodes were not generated: neither a ROCm nor a CUDA toolchain was found 1>&2
	exit /b 1
)

if not defined UNITTEST_FILTER set "UNITTEST_FILTER=-*link_bundledBc*:*VulkanComputeSimple64*"
..\dist\bin\Release\UnitTest64.exe --gtest_filter=%UNITTEST_FILTER% --gtest_output=xml:../result.xml
exit /b

rem Exit code 127 means the generator itself found no toolchain.
:gen
call ..\UnitTest\bitcodes\%1
if errorlevel 128 (
	echo warning: %1 failed 1>&2
) else if errorlevel 127 (
	rem toolchain absent
) else if errorlevel 1 (
	echo warning: %1 failed 1>&2
) else (
	set GENERATED=1
)
exit /b
