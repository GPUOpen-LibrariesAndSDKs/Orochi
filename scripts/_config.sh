# Shared configuration resolution for the unittest_*.sh runners.
# Sourced, not executed. Sets CONFIG, SUFFIX and UNITTEST_BIN from $1.
# Usage: . "$(dirname "$0")/_config.sh" "$1"

CONFIG=${1:-Release}

case "$CONFIG" in
    Debug|DebugFast)
        SUFFIX="D"
        ;;
    RelWithDebInfo|Release)
        SUFFIX=""
        ;;
    *)
        echo "error: unknown configuration '$CONFIG'" >&2
        echo "usage: [Debug|DebugFast|RelWithDebInfo|Release]" >&2
        exit 1
        ;;
esac

UNITTEST_BIN="../dist/bin/${CONFIG}/UnitTest64${SUFFIX}"
