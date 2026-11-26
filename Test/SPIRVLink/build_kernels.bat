@echo off
rem Regenerates the prebuilt SPIR-V modules in Test\SPIRVLink\spirv\ with the HIP SDK compiler (run from anywhere).
rem   lib.spv, user.spv : the two translation units as relocatable AMDGCN SPIR-V (linked at run time by the test)
rem   all.spv           : the same two units pre-linked at the LLVM level into one module (control)
rem Needs HIP_PATH to point at a HIP SDK whose clang supports --offload-arch=amdgcnspirv.
setlocal
if "%HIP_PATH%"=="" (
	echo HIP_PATH is not set
	exit /b 1
)
set "L=%HIP_PATH%\bin"
cd /d "%~dp0"
set "TMPDIR=%TEMP%\spirvlink_%RANDOM%"
mkdir "%TMPDIR%"
if not exist spirv mkdir spirv

rem The SPIR-V is emitted with the same backend step the driver uses for a non-relocatable amdgcnspirv compile.
for %%m in (lib user) do (
	"%L%\clang++.exe" -x hip --offload-arch=amdgcnspirv --cuda-device-only -fgpu-rdc -c -emit-llvm -O3 --rocm-path="%HIP_PATH%" kernels\%%m.hip -o "%TMPDIR%\%%m.bc" || goto :fail
	"%L%\clang.exe" -cc1 -triple=spirv64-amd-amdhsa -emit-obj -disable-llvm-optzns -mllvm -spirv-preserve-auxdata -x ir "%TMPDIR%\%%m.bc" -o spirv\%%m.spv || goto :fail
)
"%L%\llvm-link.exe" "%TMPDIR%\lib.bc" "%TMPDIR%\user.bc" -o "%TMPDIR%\all.bc" || goto :fail
"%L%\clang.exe" -cc1 -triple=spirv64-amd-amdhsa -emit-obj -disable-llvm-optzns -mllvm -spirv-preserve-auxdata -x ir "%TMPDIR%\all.bc" -o spirv\all.spv || goto :fail
rmdir /s /q "%TMPDIR%"
echo wrote spirv\lib.spv spirv\user.spv spirv\all.spv
exit /b 0

:fail
echo build failed
rmdir /s /q "%TMPDIR%"
exit /b 1
