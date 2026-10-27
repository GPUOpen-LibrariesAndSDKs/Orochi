#!/usr/bin/env bash
# generate-test.sh
# author: Sho Ikeda
# brief: Generation-only checks of the premake option paths; nothing is compiled.
# copyright: © 2026 Sho Ikeda
# This software is released under the MIT License.
# https://opensource.org/license/mit
#
# Generation-only checks of the premake option paths: cheap enough to run locally and in CI.
#   scripts/ci/generate-test.sh [scratch-dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

premake=tools/premake5/linux64/premake5
out="${1:-$(mktemp -d)}"
mkdir -p "$out"
gen() { "$premake" "$@" >"$out/last.log" 2>&1 || { cat "$out/last.log"; return 1; }; }
fails() { if "$premake" "$@" >"$out/last.log" 2>&1; then echo "expected failure: $*"; return 1; fi; }
check() { echo "ok: $*"; }

# --version and --help name no action.
"$premake" --version >/dev/null
help="$("$premake" --help || true)" # --help exits with 1
# A mixed-case trigger would end the listing early.
grep -q -- '--bakekernel' <<<"$help"
grep -q -- '--werror' <<<"$help"
grep -q -- '--forcecuda' <<<"$help"
check "--version / --help"

gen --builddir="$out/def" gmake
test -f "$out/def/Makefile"
check "default gmake"

gen --clang --warning=extra --werror --sanitizer=address --builddir="$out/strict" ninja
check "strict ninja"

gen --forceCuda --builddir="$out/cuda" gmake
check "--forceCuda gmake"

gen --os=windows --builddir="$out/win" vs2026
test -f "$out/win/YamatanoOrochi.slnx"
check "--os=windows vs2026"

gen --os=windows --sanitizer=address --builddir="$out/winasan" vs2026
check "--os=windows --sanitizer=address vs2026"

gen --builddir="$out/cc" compilecommands --cc-config=Debug --cc-output="$out/cc/compile_commands.json"
grep -q '"file"' "$out/cc/compile_commands.json"
check "compilecommands"

fails --clang=foo --builddir="$out/bad" gmake
fails --warning=foo --builddir="$out/bad" gmake
fails --os=windows --sanitizer=undef --builddir="$out/bad" vs2026
check "invalid options are rejected"

echo "all generation checks passed ($out)"
