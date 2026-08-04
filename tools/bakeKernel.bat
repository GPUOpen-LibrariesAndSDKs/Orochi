@echo off
REM Bake GPU kernels into header files as string literals.
REM Generates: ParallelPrimitives/cache/Kernels.h and KernelArgs.h

echo // automatically generated, don't edit > ParallelPrimitives/cache/Kernels.h
if errorlevel 1 exit /b 1
echo // automatically generated, don't edit > ParallelPrimitives/cache/KernelArgs.h
if errorlevel 1 exit /b 1

python tools/stringify.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/Kernels.h
if errorlevel 1 exit /b 1
python tools/genArgs.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/KernelArgs.h
if errorlevel 1 exit /b 1

python tools/stringify.py ./ParallelPrimitives/RadixSortConfigs.h  >> ParallelPrimitives/cache/Kernels.h
if errorlevel 1 exit /b 1
