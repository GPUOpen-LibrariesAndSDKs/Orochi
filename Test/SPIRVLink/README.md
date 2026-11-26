# SPIR-V run-time linking on the HIP backend (test)

Companion to the `SPIRV` example. Two HIP translation units are compiled to raw AMDGCN SPIR-V
(`--offload-arch=amdgcnspirv --no-gpu-bundle-output`): `kernels/lib.hip` defines `Vec3 libVec( Vec3 )`, a function that returns a
12-byte struct by value, and `kernels/user.hip` has a kernel that calls it and writes `2 10 40`. The project's
pre-build step compiles them with the HIP SDK compiler (`HIP_PATH` on Windows, `ROCM_PATH` or `/opt/rocm`
elsewhere) into `<executable dir>/SPIRVLink/`, so the modules always match the installed runtime.

The test takes no arguments and runs three cases:

0. `all.bundle`, both units compiled as one module (`kernels/all.hip`) in an offload bundle, loaded with
   `oroModuleLoadData` — shows whether the runtime JIT-compiles SPIR-V at all (what the `SPIRV` example does).
1. `lib.spv` + `user.spv` linked at run time with `hipLinkCreate` / `hipLinkAddData( hipJitInputSpirv )` /
   `hipLinkComplete`, loaded with `oroModuleLoadData` — the case under test.
2. `all.spv`, the single module again, through `hipLink*` — the control for case 1.

It exits with **100** when all cases produce the expected values and **99** otherwise, following Orochi's
demo conventions. Currently case 1 fails; depending on the compiler/runtime pair it either returns wrong values
or `hipLinkComplete` fails with `hipErrorInvalidConfiguration`. When an API call fails, the test prints the
actions comgr ran and their messages (comgr's log is redirected to `SPIRVLink_comgr.log` next to the executable
unless `AMD_COMGR_REDIRECT_LOGS` is already set):

```
$ ./dist/bin/Release/SPIRVLink64
HIP runtime: /opt/rocm/lib/libamdhip64.so.7
Case 0: all.bundle loaded with hipModuleLoadData (SPIR-V JIT, no linking)
  out = 2 10 40 (expected 2 10 40): PASS
Case 1: lib.spv + user.spv linked at run time with hipLink*
  out = 0 0 40 (expected 2 10 40): FAIL
Case 2 (control for case 1): all.spv, both units compiled as one module, via hipLink*
  out = 2 10 40 (expected 2 10 40): PASS
SPIRVLink: FAIL
```

Observed with ROCm 10.2 (HIP 7.17, comgr 3.5, AMD clang 24) on a Radeon AI PRO R9700 (gfx1201); modules
built by the ROCm 7.2.0 compiler fail case 1 on the same runtime with `hipErrorInvalidConfiguration`. On
Windows (HIP SDK 7.1 compiler, the display driver's `amdhip64_7.dll` with a clang 24.0.0git comgr, gfx1201)
cases 0 and 2 pass and case 1 fails in `hipLinkComplete` with `Error in hip linker: unable to add device libs
to linked bitcode`, i.e. in the bitcode link that follows the per-module translation.

## Build and run

Linux:

```sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64 SPIRVLink      # the pre-build step writes dist/bin/Release/SPIRVLink/{lib,user,all}.spv
./dist/bin/Release/SPIRVLink64
```

Windows: `tools\premake5\win\premake5.exe vs2022` (with `HIP_PATH` set, as the HIP SDK installer does), open
`build\YamatanoOrochi.sln`, build and run the `SPIRVLink` project. The modules are found relative to the
executable, so no working directory needs to be set. Arguments, if given, are treated as the module list of a
single custom case. Orochi does not wrap `hipLink*` yet; the test resolves the four entry points from the HIP
runtime that hipew loaded.

## Cause

The SPIR-V is correct: in `user.spv` the imported `libVec` has the same function type as its definition in
`lib.spv` (`OpTypeFunction` returning `OpTypeStruct { float, float, float }`). The mismatch is introduced
when comgr translates each module back to LLVM IR separately (`amd-llvm-spirv -r --spirv-target-env=CL2.0
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
uninitialised `sret` slot and passes its arguments shifted by one. A single module is unaffected because every
struct-returning function in it has a body. `postProcessBuiltinsWithArrayArguments()`
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
