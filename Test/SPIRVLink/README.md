# SPIR-V run-time linking on the HIP backend (test)

Companion to the `SPIRV` example. Two HIP translation units are compiled to relocatable AMDGCN SPIR-V
(`--offload-arch=amdgcnspirv -fgpu-rdc`): `kernels/lib.hip` defines `Vec3 libVec( Vec3 )`, a function that
returns a 12-byte struct by value, and `kernels/user.hip` has a kernel that calls it and writes `2 10 40`.
The prebuilt modules are committed in `spirv/`, so the test needs no compiler at run time.

The test takes no arguments and runs two cases:

1. `lib.spv` + `user.spv` linked at run time with `hipLinkCreate` / `hipLinkAddData( hipJitInputSpirv )` /
   `hipLinkComplete`, loaded with `oroModuleLoadData` — the case under test.
2. `all.spv`, the same two units pre-linked into one SPIR-V module — the control.

It exits with **100** when both cases produce the expected values and **99** otherwise, following Orochi's
demo conventions. Currently case 1 fails:

```
$ ./dist/bin/Release/SPIRVLink64
Case 1: lib.spv + user.spv linked at run time with hipLink*
  out = 0 0 40 (expected 2 10 40): FAIL
Case 2 (control): all.spv, the same code pre-linked into one module
  out = 2 10 40 (expected 2 10 40): PASS
SPIRVLink: FAIL
```

Observed with ROCm 10.2 (HIP 7.17, comgr 3.5, AMD clang 24) on a Radeon AI PRO R9700 (gfx1201).

## Build and run

Linux:

```sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64 SPIRVLink
./dist/bin/Release/SPIRVLink64          # from the repository root; the modules are found relative to it or to the executable
```

Windows: `tools\premake5\win\premake5.exe vs2022`, open `build\YamatanoOrochi.sln`, build and run the
`SPIRVLink` project (`dist\bin\Release\SPIRVLink64.exe`; the modules are found relative to the executable, so
no working directory needs to be set).

Arguments, if given, are treated as the module list of a single custom case.

The committed modules were emitted by ROCm 10.2's compiler and declare the `SPV_KHR_untyped_pointers` and
`SPV_AMD_weak_linkage` extensions. If the runtime cannot consume them, `hipLinkComplete` fails with
`hipErrorInvalidConfiguration` for both cases; regenerate the modules with the compiler that matches the runtime:
`build_kernels.sh` on Linux (`ROCM_PATH` defaults to `/opt/rocm`; it also prints the translated signatures
shown below), `build_kernels.bat` on Windows (`HIP_PATH` must point at the HIP SDK). To see where comgr fails,
set `AMD_COMGR_EMIT_VERBOSE_LOGS=1` and `AMD_COMGR_REDIRECT_LOGS=stdout` (and `AMD_LOG_LEVEL=4` for the HIP
runtime) before running. Orochi does not wrap `hipLink*` yet; the test resolves the four entry points from the
HIP runtime that hipew loaded.

## Cause

The SPIR-V is correct: in `user.spv` the imported `libVec` has the same function type as its definition in
`lib.spv` (`OpTypeFunction` returning `OpTypeStruct { float, float, float }`). The mismatch is introduced when
comgr translates each module back to LLVM IR separately (`amd-llvm-spirv -r --spirv-target-env=CL2.0
--spirv-preserve-auxdata`, action `AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC`) before linking them:

```
user module: declare spir_func void @_Z6libVec4Vec3(ptr sret(%struct.Vec3), float, float, float)
lib module:  define spir_func %struct.Vec3 @_Z6libVec4Vec3(float %0, float %1, float %2) #4
```

In the translator (ROCm/SPIRV-LLVM-Translator, amd-staging, `f3fe1819845c`), `SPIRVToLLVM::translate()`
(`lib/SPIRV/SPIRVReader.cpp`, near line 4746) calls `postProcessBuiltinsReturningStruct()`
(`lib/SPIRV/SPIRVUtil.cpp:2508`). That pass takes every function *declaration* whose return type is a struct
and whose name passes `oclIsBuiltin()` for an OpenCL builtin and rewrites it, together with its call sites, to
`void f( ptr sret(T), ... )`. `oclIsBuiltin()` accepts any Itanium-mangled global-namespace function, so an
imported HIP function qualifies; definitions are never rewritten. Caller and callee therefore disagree on the
ABI as soon as they live in different modules: the callee returns in registers while the caller reads an
uninitialised `sret` slot and passes its arguments shifted by one. A single pre-linked module is unaffected
because every struct-returning function in it has a body. `postProcessBuiltinsWithArrayArguments()`
(`lib/SPIRV/SPIRVToOCL12.cpp:62-63`, `SPIRVToOCL20.cpp:62-63`) does the same to by-value array parameters of
declarations.

Which declarations are rewritten (translator only, independent of `--spirv-target-env`):

| declaration | mangled name | after `amd-llvm-spirv -r` |
|---|---|---|
| `Vec3 freeFn( Vec3 )` | `_Z6freeFn4Vec3` | `void ( ptr sret(Vec3), float, float, float )` |
| `Vec3 freeFnNoArg()` | `_Z11freeFnNoArgv` | `void ( ptr sret(Vec3) )` |
| `Vec3 rt::nsFn( Vec3 )` | `_ZN2rt4nsFnE4Vec3` | unchanged |
| `Vec3 Cls::member( Vec3 )` | `_ZN3Cls6memberE4Vec3` | unchanged |
| `extern "C" Vec3 cFn( Vec3 )` | `cFn` | unchanged |
