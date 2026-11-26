# SPIR-V run-time linking on the HIP backend (reproducer)

Companion to the `SPIRV` example. Two HIP translation units are compiled to relocatable AMDGCN SPIR-V
(`--offload-arch=amdgcnspirv -fgpu-rdc`): `lib.hip` defines `Vec3 libVec( Vec3 )`, a function that returns a
12-byte struct by value, and `user.hip` has a kernel that calls it. The demo links the modules at run time
with `hipLinkCreate` / `hipLinkAddData( hipJitInputSpirv )` / `hipLinkComplete`, loads the result with
`oroModuleLoadData` and runs the kernel. Expected output is `2 10 40`.

With the two modules linked at run time the struct result is lost. The same two units pre-linked into one
SPIR-V module give the right answer:

```
$ ./dist/bin/Release/SPIRVLink64 dist/SPIRVLink/lib.spv dist/SPIRVLink/user.spv
2 module(s): out = 0 0 40 (expected 2 10 40): FAIL
$ ./dist/bin/Release/SPIRVLink64 dist/SPIRVLink/all.spv
1 module(s): out = 2 10 40 (expected 2 10 40): PASS
```

Observed with ROCm 10.2 (HIP 7.17, comgr 3.5, AMD clang 24) on a Radeon AI PRO R9700 (gfx1201).

## Build and run (Linux)

```sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64 SPIRVLink

./Test/SPIRVLink/build_kernels.sh          # writes dist/SPIRVLink/{lib,user,all}.spv

AMD_COMGR_CACHE=0 ./dist/bin/Release/SPIRVLink64 dist/SPIRVLink/lib.spv dist/SPIRVLink/user.spv   # wrong
AMD_COMGR_CACHE=0 ./dist/bin/Release/SPIRVLink64 dist/SPIRVLink/all.spv                            # correct
```

The demo exits with **100** when the kernel result is correct and **99** otherwise, following Orochi's demo
conventions, so the two-module run exits with 99 until the issue below is fixed. Orochi does not wrap
`hipLink*` yet; the demo resolves the four entry points from the HIP runtime that hipew loaded (Linux only).

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

`build_kernels.sh` prints the two translated signatures above when `amd-llvm-spirv` is available.
