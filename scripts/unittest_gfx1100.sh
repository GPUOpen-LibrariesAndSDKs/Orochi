#!/bin/sh
export OROCHI_OFFLOAD_ARCH=gfx1100
UNITTEST_FILTER='-*link*:*getErrorString*' exec "$(dirname "$0")/unittest.sh"
