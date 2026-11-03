#!/bin/sh
# Bundled LLVM bitcode for the link_bundledBc* tests (hiprtc link input).
set -e
cd "$(dirname "$0")"

ROCM="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
# Older HIP SDKs ship only hipcc, which takes the same flags.
for CXX in "$ROCM/lib/llvm/bin/amdclang++" "$ROCM/bin/amdclang++" "$ROCM/llvm/bin/amdclang++" "$ROCM/bin/hipcc"; do
	[ -x "$CXX" ] && break
done
# pip-installed ROCm (TheRock) keeps the device libraries only under lib/llvm.
DEVLIB=""
if [ ! -d "$ROCM/amdgcn/bitcode" ] && [ -d "$ROCM/lib/llvm/amdgcn/bitcode" ]; then
	DEVLIB="--hip-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode"
fi
# The test device's arch must be in the bundle. Override with e.g. OROCHI_OFFLOAD_ARCH=native.
ARCH="${OROCHI_OFFLOAD_ARCH:-gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1011,gfx1012,gfx1013,gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1150,gfx1151,gfx1200,gfx1201}"

# --no-offload-new-driver: the new driver emits an LLVM offload binary that orortcLink cannot read.
# -cuid: a fixed id instead of a random one keeps the output reproducible.
# No --offload-compress: orortcLink rejects compressed (CCOB) bundles.
for SRC in moduleTestKernel moduleTestFunc; do
	"$CXX" -x hip -O3 --no-offload-new-driver --cuda-device-only -fgpu-rdc -c -emit-llvm --gpu-bundle-output \
		--offload-arch="$ARCH" -cuid="$SRC" $DEVLIB "../$SRC.cpp" -o "$SRC-hip-amdgcn-amd-amdhsa.bc"
done
