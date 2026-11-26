#!/usr/bin/env bash
# Regenerates the prebuilt SPIR-V modules in Test/SPIRVLink/spirv/ (run from anywhere).
#   lib.spv, user.spv : the two translation units as relocatable AMDGCN SPIR-V (linked at run time by the test)
#   all.spv           : the same two units pre-linked at the LLVM level into one module (control)
# Requires a ROCm compiler with amdgcnspirv support (the committed files were built with ROCm 10.2).
set -eu
ROCM=${ROCM_PATH:-/opt/rocm}
L=$ROCM/llvm/bin
cd "$(dirname "$0")"
SRC=kernels
OUT=spirv
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

# `clang++ --hip-link --cuda-device-only --offload-arch=amdgcnspirv` does not link relocatable inputs, so the
# SPIR-V is emitted with the same backend step the driver uses for a non-relocatable amdgcnspirv compile.
emit_spirv() { # <bitcode> <spirv>
	"$L/clang" -cc1 -triple=spirv64-amd-amdhsa -emit-obj -disable-llvm-optzns -mllvm -spirv-preserve-auxdata -x ir "$1" -o "$2"
}

for m in lib user; do
	"$L/clang++" -x hip --offload-arch=amdgcnspirv --cuda-device-only -fgpu-rdc -c -emit-llvm -O3 "$SRC/$m.hip" -o "$TMP/$m.bc"
	emit_spirv "$TMP/$m.bc" "$OUT/$m.spv"
done
"$L/llvm-link" "$TMP/lib.bc" "$TMP/user.bc" -o "$TMP/all.bc"
emit_spirv "$TMP/all.bc" "$OUT/all.spv"

# Show the cause: translating the modules back to LLVM the way comgr does rewrites the DECLARATION of libVec
# (user module) to an sret convention, while the DEFINITION (lib module) keeps returning the struct by value.
if [ -x "$L/amd-llvm-spirv" ]; then
	for m in lib user; do
		"$L/amd-llvm-spirv" -r --spirv-target-env=CL2.0 --spirv-preserve-auxdata "$OUT/$m.spv" -o "$TMP/$m.rt.bc"
	done
	echo "libVec after SPIR-V -> LLVM translation:"
	"$L/llvm-dis" "$TMP/user.rt.bc" -o - | grep -E '^declare .*libVec' | sed 's/^/  user module: /'
	"$L/llvm-dis" "$TMP/lib.rt.bc" -o - | grep -E '^define .*libVec' | sed 's/ {$//; s/^/  lib module:  /'
fi
echo "wrote $OUT/lib.spv $OUT/user.spv $OUT/all.spv"
