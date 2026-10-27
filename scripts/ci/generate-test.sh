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
# Whether the <config>|x64 settings of a vcxproj contain a pattern. Not grep -q: under pipefail its
# early exit can fail the pipeline with SIGPIPE.
insection() { awk "/<ItemDefinitionGroup .*'$2\|x64'\">/,/<\/ItemDefinitionGroup>/" "$1" | grep -- "$3" >/dev/null; }

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
grep -q 'libOrochi64D.a' "$out/def/UnitTest/Makefile"
grep -q -- '--start-group' "$out/def/UnitTest/Makefile"
check "default gmake"

# Strict flags apply to first-party code only; projects tagged thirdparty compile without warnings.
gen --clang --warning=extra --werror --sanitizer=address --builddir="$out/strict" ninja
grep -q 'fsanitize=address' "$out/strict/Orochi/Orochi.ninja"
grep -q 'Werror' "$out/strict/Orochi/Orochi.ninja"
if grep -q 'Werror' "$out/strict/contrib/cuew/cuew.ninja"; then echo "cuew got -Werror"; exit 1; fi
grep -q 'fuse-ld=lld' "$out/strict/UnitTest/UnitTest.ninja"
check "strict ninja"

gen --bakeKernel --builddir="$out/bake" gmake
# Option lookups ignore case: --bakeKernel reaches both _OPTIONS and the options: filter.
grep -q 'Baking ParallelPrimitives kernels' "$out/bake/ParallelPrimitives/Makefile"
grep -q 'ORO_PP_LOAD_FROM_STRING' "$out/bake/UnitTest/Makefile"
gen --forceCuda --builddir="$out/cuda" gmake
grep -q 'OROCHI_ENABLE_CUEW' "$out/cuda/Orochi/Makefile"
check "--bakeKernel / --forceCuda gmake"

gen --os=windows --builddir="$out/win" vs2026
test -f "$out/win/YamatanoOrochi.slnx"
# DebugFast is optimized, so MSBuild's default /RTC1 for the debug CRT must be off (D8016).
insection "$out/win/Orochi/Orochi.vcxproj" DebugFast '<BasicRuntimeChecks>Default'
insection "$out/win/Orochi/Orochi.vcxproj" DebugFast '<InlineFunctionExpansion>OnlyExplicitInline'
# The runtime DLLs are copied under their own names; a bare directory would become xcopy's "dir*".
grep -q 'hiprtc0701.dll\*"' "$out/win/Orochi/Orochi.vcxproj"
if grep -q 'Debug\*"' "$out/win/Orochi/Orochi.vcxproj"; then echo "DLLs copied to a file named Debug"; exit 1; fi
check "--os=windows vs2026"

# MSVC's AddressSanitizer rejects Edit and Continue.
gen --os=windows --sanitizer=address --builddir="$out/winasan" vs2026
insection "$out/winasan/Orochi/Orochi.vcxproj" Debug '<DebugInformationFormat>ProgramDatabase'
check "--os=windows --sanitizer=address vs2026"

# Apple ld rejects --start-group.
gen --os=macosx --builddir="$out/mac" gmake
if grep -rq -- '--start-group' "$out/mac"; then echo "--start-group on macOS"; exit 1; fi
check "--os=macosx gmake"

gen --builddir="$out/cc" compilecommands --cc-config=Debug --cc-output="$out/cc/compile_commands.json"
grep -q '"file"' "$out/cc/compile_commands.json"
check "compilecommands"

fails --clang=foo --builddir="$out/bad" gmake
fails --warning=foo --builddir="$out/bad" gmake
fails --os=windows --sanitizer=undef --builddir="$out/bad" vs2026
check "invalid options are rejected"

echo "all generation checks passed ($out)"
