#!/bin/sh
# Bake GPU kernels into header files as string literals.
# Generates: ParallelPrimitives/cache/Kernels.h and KernelArgs.h

echo "// automatically generated, don't edit" > ParallelPrimitives/cache/Kernels.h
echo "// automatically generated, don't edit" > ParallelPrimitives/cache/KernelArgs.h

python tools/stringify.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/Kernels.h
python tools/genArgs.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/KernelArgs.h

python tools/stringify.py ./ParallelPrimitives/RadixSortConfigs.h  >> ParallelPrimitives/cache/Kernels.h
