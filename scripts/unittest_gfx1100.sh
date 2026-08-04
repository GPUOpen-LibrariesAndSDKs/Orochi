#!/bin/sh
# Run Orochi unit tests for gfx1100 (Linux)
# Usage: ./unittest_gfx1100.sh [Debug|DebugFast|RelWithDebInfo|Release]

set -e

# Every path below is relative to this script's directory.
cd "$(dirname "$0")"
. ./_config.sh "$1"

rm -rf cache
cd ../UnitTest/bitcodes && ./generate_bitcodes_gfx1100.sh
cd ../../scripts
"${UNITTEST_BIN}" --gtest_filter=-*link*:*getErrorString* --gtest_output=xml:../result.xml
