#!/bin/sh
# Relocatable fatbins for the link_bundledBc* tests on CUDA.
set -e
cd "$(dirname "$0")"

NVCC="${CUDA_PATH:+$CUDA_PATH/bin/}nvcc"
for SRC in moduleTestFunc moduleTestKernel; do
	"$NVCC" -x cu -fatbin --device-c -arch=all --threads 4 "../$SRC.cpp" -o "$SRC.fatbin"
done
