#!/usr/bin/env bash
# Builds the SPIR-V modules for the SPIRVLink demo into dist/SPIRVLink/ (run from anywhere).
#   lib.spv, user.spv : the two translation units as relocatable AMDGCN SPIR-V (linked at run time by the demo)
#   all.spv           : the same two units pre-linked at the LLVM level into one module (control)
# Requires a ROCm compiler with amdgcnspirv support (tested with ROCm 10.2).
set -eu
ROCM=${ROCM_PATH:-/opt/rocm}
L=$ROCM/llvm/bin
cd "$(dirname "$0")/../.."
SRC=Test/SPIRVLink/kernels
OUT=dist/SPIRVLink
mkdir -p "$OUT"

# `clang++ --hip-link --cuda-device-only --offload-arch=amdgcnspirv` does not link relocatable inputs, so the
# SPIR-V is emitted with the same backend step the driver uses for a non-relocatable amdgcnspirv compile.
emit_spirv() { # <bitcode> <spirv>
	"$L/clang" -cc1 -triple=spirv64-amd-amdhsa -emit-obj -disable-llvm-optzns -mllvm -spirv-preserve-auxdata -x ir "$1" -o "$2"
}

for m in lib user; do
	"$L/clang++" -x hip --offload-arch=amdgcnspirv --cuda-device-only -fgpu-rdc -c -emit-llvm -O3 "$SRC/$m.hip" -o "$OUT/$m.bc"
	emit_spirv "$OUT/$m.bc" "$OUT/$m.spv"
done
"$L/llvm-link" "$OUT/lib.bc" "$OUT/user.bc" -o "$OUT/all.bc"
emit_spirv "$OUT/all.bc" "$OUT/all.spv"

# Show the cause: translating the modules back to LLVM the way comgr does rewrites the DECLARATION of libVec
# (user module) to an sret convention, while the DEFINITION (lib module) keeps returning the struct by value.
if [ -x "$L/amd-llvm-spirv" ]; then
	for m in lib user; do
		"$L/amd-llvm-spirv" -r --spirv-target-env=CL2.0 --spirv-preserve-auxdata "$OUT/$m.spv" -o "$OUT/$m.rt.bc"
	done
	echo "libVec after SPIR-V -> LLVM translation:"
	"$L/llvm-dis" "$OUT/user.rt.bc" -o - | grep -E '^declare .*libVec' | sed 's/^/  user module: /'
	"$L/llvm-dis" "$OUT/lib.rt.bc" -o - | grep -E '^define .*libVec' | sed 's/ {$//; s/^/  lib module:  /'
fi
echo "wrote $OUT/lib.spv $OUT/user.spv $OUT/all.spv"
