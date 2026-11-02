@echo off
rem Bundled LLVM bitcode for the link_bundledBc* tests (hiprtc link input).
setlocal
cd /d "%~dp0"

if defined ROCM_PATH (set "ROCM=%ROCM_PATH%") else (set "ROCM=%HIP_PATH%")
set "CXX=%ROCM%\lib\llvm\bin\amdclang++.exe"
if not exist "%CXX%" set "CXX=%ROCM%\bin\amdclang++.exe"
rem Older HIP SDKs ship only clang++.exe, the same driver under another name.
if not exist "%CXX%" set "CXX=%ROCM%\bin\clang++.exe"
set "DEVLIB="
if not exist "%ROCM%\amdgcn\bitcode" if exist "%ROCM%\lib\llvm\amdgcn\bitcode" set DEVLIB="--hip-device-lib-path=%ROCM%\lib\llvm\amdgcn\bitcode"
rem The test device's arch must be in the bundle. Override with e.g. OROCHI_OFFLOAD_ARCH=native.
if not defined OROCHI_OFFLOAD_ARCH set "OROCHI_OFFLOAD_ARCH=gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1011,gfx1012,gfx1013,gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1150,gfx1151,gfx1200,gfx1201"

rem See generate_bitcodes.sh for why each flag is needed.
for %%S in (moduleTestKernel moduleTestFunc) do (
	"%CXX%" -x hip -O3 --no-offload-new-driver --cuda-device-only -fgpu-rdc -c -emit-llvm --gpu-bundle-output --offload-arch=%OROCHI_OFFLOAD_ARCH% -cuid=%%S %DEVLIB% ../%%S.cpp -o %%S-hip-amdgcn-amd-amdhsa.bc || exit /b 1
)
