#!/bin/sh
# Runs UnitTest64; the tests open sources by "../UnitTest/..." relative to scripts/.
# UNITTEST_FILTER overrides the gtest filter, OROCHI_OFFLOAD_ARCH the bitcode arch list.
set -eu
cd "$(dirname "$0")"

rm -rf cache

have_rocm() {
	rocm="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
	for c in lib/llvm/bin/amdclang++ bin/amdclang++ llvm/bin/amdclang++ bin/hipcc; do
		[ -x "$rocm/$c" ] && return 0
	done
	return 1
}
have_nvcc() { command -v "${CUDA_PATH:+$CUDA_PATH/bin/}nvcc" >/dev/null 2>&1; }

# A generator whose toolchain is absent is skipped; any other failure only warns.
generated=0
for GEN in generate_bitcodes.sh generate_bitcodes_nvidia.sh; do
	case $GEN in
		generate_bitcodes.sh) have=have_rocm; what=ROCm ;;
		*) have=have_nvcc; what=nvcc ;;
	esac
	if ! $have; then
		echo "note: skipping $GEN: $what not found" >&2
		continue
	fi
	rc=0
	../UnitTest/bitcodes/$GEN || rc=$?
	case $rc in
		0) generated=1 ;;
		127) ;;
		*) echo "warning: $GEN failed (exit $rc)" >&2 ;;
	esac
done
if [ "$generated" = 0 ]; then
	echo "error: bitcodes were not generated: neither a ROCm nor a CUDA toolchain was found" >&2
	exit 1
fi

../dist/bin/Release/UnitTest64 --gtest_filter="${UNITTEST_FILTER:--*link_bundledBc*}" --gtest_output=xml:../result.xml
