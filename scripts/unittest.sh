#!/bin/sh
# Run Orochi unit tests (Linux)
# Cleans cache, generates bitcodes, and runs the test suite.
# Usage: ./unittest.sh [Debug|DebugFast|RelWithDebInfo|Release]

set -e

# Every path below is relative to this script's directory.
cd "$(dirname "$0")"
. ./_config.sh "$1"

rm -rf cache
cd ../UnitTest/bitcodes && ./generate_bitcodes.sh
cd ../../scripts
"${UNITTEST_BIN}" --gtest_filter=-*link_bundledBc* --gtest_output=xml:../result.xml
