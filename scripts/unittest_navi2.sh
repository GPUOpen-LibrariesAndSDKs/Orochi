#!/bin/sh
UNITTEST_FILTER='-*link*:*getErrorString*' exec "$(dirname "$0")/unittest.sh"
