#!/bin/sh
# Relocatable fatbins for the link_bundledBc* tests on CUDA.
set -e
cd "$(dirname "$0")"

NVCC="${CUDA_PATH:+$CUDA_PATH/bin/}nvcc"
if ! command -v "$NVCC" >/dev/null 2>&1; then
	echo "error: nvcc not found (set CUDA_PATH)" >&2
	exit 127
fi
for SRC in moduleTestFunc moduleTestKernel; do
	"$NVCC" -x cu -fatbin --device-c -arch=all --threads 4 "../$SRC.cpp" -o "$SRC.fatbin"
done
