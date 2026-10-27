
## 2026-10-24 — hipew: generated library search lists and ORO_AMDHIP64_PATH / ORO_HIPRTC_PATH

- `contrib/hipew/src/hipew.cpp`: the hand-written `amdhip64` / `hiprtc` path arrays are now built by `getHipBuiltinPaths()` / `getHiprtcBuiltinPaths()`.
  Windows uses a (major, max minor, min minor) table; Linux globs `/opt/rocm/core-*` newest first (so `core-10.2` etc. need no edit), then the traditional layout, then the generic symlink.
- Search order without `customPaths_*`: `ORO_AMDHIP64_PATH` / `ORO_HIPRTC_PATH` (`:`-separated, `;` on Windows) → built-in list. Non-loadable env entries print one stderr line with `dlerror()` and fall through. `customPaths_*` still replaces everything. `secure_getenv` is used on glibc.
- Windows legacy `hiprtc070NN.dll` spelling is also tried for the modern name 7.10–7.14 (the old list had it only for some); harmless extra probes.
- Fixes: `hip_lib == NULL` now returns `HIPEW_ERROR_OPEN_FAILED` (was `ATEXIT_FAILED`). The Windows old-driver check now runs on the loaded DLL (`GetModuleFileNameA`) instead of `hip_paths[0]`; hiprtc is loaded after it. **Windows code is not compiled or run here.**
- Checked on Linux (premake ninja --clang, Release, ROCm core-10.1 + core-10 installed): default loads `core-10.1/lib/libamdhip64.so.7`; env pointing to `core-10/lib/libamdhip64.so.7` + `libhiprtc.so.7` loads those; a bad entry warns and falls back; `DeviceEnum64` reports 3 AMD devices in all cases.
- The edit hook's clang-format reflowed the whole file (~1300 lines); edits were applied through a script to keep the diff to the touched regions.

## 2026-10-25 — premake 5.0.0 review items (§2.1–2.3, 3.2–3.6, 4)

Source: `tmp/2026-10-25-premake-5-0-0-build-review.md`. §2.2 took option 1 (no LTO anywhere); §3.1 was not requested and the default builddir is still `.`.

- MSVC DebugFast: `optimize "Speed"` + `/Ob1` + `runtimechecks "Off"`; `editandcontinue` On only for Debug (and Off under ASan). Verified in generated vcxproj only (not built on Windows).
- Release: `optimize "Speed"`, LTO filter and `/favor:AMD64` removed. `intrinsics "On"` kept.
- `thirdparty` tag on cuew/hipew/gtest/stb; one workspace filter sets `warnings "Off"`, `removefatalwarnings`, and turns conformance off for MSVC. vcxproj of gtest shows `ConformanceMode=false`; gcc ninja flags for cuew show `-w`.
- `bakeKernel` is now a custom build rule on `RadixSortKernels.h` (input `RadixSortConfigs.h`, outputs `cache/Kernels.h`, `cache/KernelArgs.h`); ninja makes them implicit deps of RadixSort.o. `tools/bakeKernel.sh` calls `python3` (the .bat still calls `python`). Not run end to end, generation only.
- Windows DLL staging uses `{COPYFILEIFNEWER}` per DLL with `%{!cfg.targetdir}`; plain `%{cfg.targetdir}` inside `%[]` yields a wrong relative path. `multiprocessorcompile` moved to workspace scope.
- `linkWin32SystemLibs` is kept: whether gmake/ninja + msc needs it is unverified.
- Removed `_DEBUGFAST`; gtest's links/include/define moved to a `usage "INTERFACE"`; ParallelPrimitives' `uses { "Orochi" }` moved to `usage "PUBLIC"`.
- New: `--werror`, `--sanitizer`, switch-value validation, `_GLIBCXX_ASSERTIONS` (Linux Debug/DebugFast), `-fcolor-diagnostics` (clang toolset), CI generate-test step. `.vscode` compile_commands task now uses `compilecommands` (16 entries for Debug; compiler name is always `clang++`).
- Measured (this machine, gcc 13.3 / clang 23.1.4, ninja, `--warning=extra --precompiled`): all 8 toolchain×config builds succeed; UnitTest 14/15 each, the failure being `OroTestBase.checkCUEW` (no CUDA SDK), same as before.

## 2026-10-27 — DLL staging target, macOS link groups, --help, CI generate test

From `FireRender/tmp/2026-10-27-premake-cross-project-review.md` (R-ORO-1, 2, 4, 5, 6 and S-1; R-ORO-3 was not requested, so the default builddir is still `.` and static libraries still go to `dist/`).

- DLL staging: `%[%{!cfg.targetdir}]` expanded to `xcopy /D /Y "src" "…\dist\bin\Debug*"`. xcopy reads a destination ending in `*` as a file name, so the DLLs did not land next to the binaries; Vkpt's Windows CI, which includes this Orochi, logs `1 File(s) copied` for each and passes. With a trailing "/" premake appends the file name (`…\Debug\hiprtc0701.dll*`). The copy is now `orochiStageRuntimeDlls()`, called by the Orochi project only in this workspace (`isOrochiWorkspace`); a host workspace keeps its libraries elsewhere and can call it from its own applications. Not run on Windows.
- `linkgroups "On"` only for `system:not macosx`: the generated macOS gmake files had `-Wl,--start-group`, which Apple ld rejects.
- `--help` listed none of the workspace options: premake's help stops at a mixed-case trigger, and `bakeKernel` sorts first. The triggers are now `bakekernel` and `forcecuda` (also in `enable_cuew.lua`); `--bakeKernel`, `--forceCuda`, `_OPTIONS["bakeKernel"]` and `filter "options:bakeKernel"` keep working (the generate test checks `ORO_PP_LOAD_FROM_STRING` and `OROCHI_ENABLE_CUEW`).
- `--warning` takes on/off/extra/full like Vkpt and Rtb (default on = premake `Default`). `main` has no `--warning`, so no released interface changes; `high` has no counterpart and is gone.
- `require("premake", ">=5.0.0")`, `premake.error` instead of `error(string.format(...))`, `inlining "Explicit"` instead of `/Ob1`.
- `stb` has an `INTERFACE` usage and Texture `uses` it; the link line moves stb after Orochi, inside `--start-group` on Linux.
- CI: the generation checks moved to `scripts/ci/generate-test.sh` (help listing, strict ninja, bake and CUDA options, vs2026 DebugFast `BasicRuntimeChecks`/`InlineFunctionExpansion`, DLL copy names, ASan without Edit and Continue, no `--start-group` on macOS, compilecommands, rejected values). The workflow uses `actions/checkout@v5`, runs on `pull_request` too, and has a Windows job (vs2022, MSBuild Release; GitHub-hosted runners ship VS 2022). The Windows job has not run yet.
- `tools/premake5/osx/premake5` is executable now (it was tracked as 100644).
- `linkWin32SystemLibs` is kept: whether gmake/ninja + msc needs it is still unverified.
- Measured (this machine): `scripts/ci/generate-test.sh` passes; gcc and clang ninja Release builds succeed with `--warning=extra`; UnitTest 14/15, the failure being `OroTestBase.checkCUEW` (no CUDA SDK), as before.
