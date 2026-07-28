#!/bin/sh
# Run Orochi unit tests (Linux)
# Cleans cache, generates bitcodes, and runs the test suite.
# Usage: ./unittest.sh [Release|Debug|DebugFast|RelWithDebInfo]

set -e

CONFIG=${1:-Release}
SUFFIX=""
if [ "$CONFIG" = "Debug" ] || [ "$CONFIG" = "DebugFast" ]; then
    SUFFIX="D"
fi

rm -rf cache
cd ../UnitTest/bitcodes && ./generate_bitcodes.sh
cd ../../scripts
../dist/bin/${CONFIG}/UnitTest64${SUFFIX} --gtest_filter=-*link_bundledBc* --gtest_output=xml:../result.xml
