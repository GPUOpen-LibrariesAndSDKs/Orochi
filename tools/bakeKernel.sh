# mkdir hiprt/cache/
echo "// automatically generated, don't edit" > ParallelPrimitives/cache/Kernels.h
echo "// automatically generated, don't edit" > ParallelPrimitives/cache/KernelArgs.h
python3 tools/stringify.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/Kernels.h
python3 tools/genArgs.py ./ParallelPrimitives/RadixSortKernels.h  >> ParallelPrimitives/cache/KernelArgs.h

python3 tools/stringify.py ./ParallelPrimitives/RadixSortConfigs.h  >> ParallelPrimitives/cache/Kernels.h
