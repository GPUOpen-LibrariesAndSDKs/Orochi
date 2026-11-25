@echo off
rem Relocatable fatbins for the link_bundledBc* tests on CUDA.
setlocal
cd /d "%~dp0"

set "NVCC=nvcc"
if defined CUDA_PATH set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
for %%S in (moduleTestFunc moduleTestKernel) do (
	"%NVCC%" -x cu -fatbin --device-c -arch=all --threads 4 ../%%S.cpp -o %%S.fatbin || exit /b 1
)
