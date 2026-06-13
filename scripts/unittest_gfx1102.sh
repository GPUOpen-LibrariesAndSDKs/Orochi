#!/bin/sh
# Run Orochi unit tests for gfx1102 (Linux)
# Usage: ./unittest_gfx1102.sh [Release|Debug|RelWithDebInfo]

set -e

CONFIG=${1:-Release}
SUFFIX=""
if [ "$CONFIG" = "Debug" ]; then
    SUFFIX="D"
fi

rm -rf cache
cd ../UnitTest/bitcodes && ./generate_bitcodes_gfx1102.sh
cd ../../scripts
../dist/bin/${CONFIG}/UnitTest64${SUFFIX} --gtest_filter=-*link*:*getErrorString* --gtest_output=xml:../result.xml
