@echo off
rem Relocatable fatbins for the link_bundledBc* tests on CUDA.
setlocal
cd /d "%~dp0"

set "NVCC=nvcc"
if defined CUDA_PATH set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
where "%NVCC%" >nul 2>&1 || if not exist "%NVCC%" (
	echo error: nvcc not found ^(set CUDA_PATH^) 1>&2
	exit /b 127
)
for %%S in (moduleTestFunc moduleTestKernel) do (
	"%NVCC%" -x cu -fatbin --device-c -arch=all --threads 4 ../%%S.cpp -o %%S.fatbin || exit /b 1
)
