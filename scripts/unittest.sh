#!/bin/sh
# Runs UnitTest64; the tests open sources by "../UnitTest/..." relative to scripts/.
# UNITTEST_FILTER overrides the gtest filter, OROCHI_OFFLOAD_ARCH the bitcode arch list.
set -eu
cd "$(dirname "$0")"

rm -rf cache

# A generator exits 127 when its toolchain is absent; any other failure aborts.
generated=0
for GEN in generate_bitcodes.sh generate_bitcodes_nvidia.sh; do
	rc=0
	../UnitTest/bitcodes/$GEN || rc=$?
	case $rc in
		0) generated=1 ;;
		127) ;;
		*) echo "error: $GEN failed (exit $rc)" >&2; exit "$rc" ;;
	esac
done
if [ "$generated" = 0 ]; then
	echo "error: bitcodes were not generated: neither a ROCm nor a CUDA toolchain was found" >&2
	exit 1
fi

../dist/bin/Release/UnitTest64 --gtest_filter="${UNITTEST_FILTER:--*link_bundledBc*}" --gtest_output=xml:../result.xml
