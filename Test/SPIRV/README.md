# SPIR-V on the HIP backend

This example loads a precompiled AMDGCN SPIR-V offload bundle through Orochi's
HIP backend. HIP JIT-compiles the device code for the selected AMD GPU. The
existing `oroModuleLoadData`, `oroModuleGetFunction`, and `oroModuleLaunchKernel`
APIs are sufficient; the example initializes only `ORO_API_HIPDRIVER` and does not
use hipRTC.

The kernel applies `value = 3 * value + 7` to 257 integers. The host checks every
result, including the final partially filled block. The device source is compiled
for `amdgcnspirv`, with no concrete `gfx` target specified.

## Linux

Run these commands from the repository root. Building the host uses Orochi's
bundled headers and dynamic loader. Compiling the device code requires a ROCm
compiler with `amdgcnspirv` support; running it requires a HIP runtime supporting
SPIR-V JIT compilation and an AMD GPU supported by that runtime.

```sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64 SPIRV

/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=amdgcnspirv \
    --cuda-device-only -O2 Test/SPIRV/kernel.hip -o dist/transform.bundle

AMD_COMGR_CACHE=0 ./dist/bin/Release/SPIRV64 dist/transform.bundle \
    /opt/rocm/lib/libamdhip64.so
```

Replace `/opt/rocm` with the installation to test. The optional final argument
selects the HIP runtime library via `oroInitialize`'s custom library search path.
Omit it to use Orochi's default search paths. `AMD_COMGR_CACHE=0` disables the
Comgr cache for this run so the example exercises JIT compilation.

The example uses device 0; `HIP_VISIBLE_DEVICES` can select a different GPU.
It prints the device name, architecture, runtime version, and
`PASS: all 257 results match`. Like the other Orochi demos, its success exit code
is **100**, and its failure exit code is **99**.

## Windows

The following commands are provided for follow-up validation; Windows execution
has not been validated for this example. Use a Visual Studio developer command
prompt with the ROCm compiler on `PATH`:

```bat
tools\premake5\win\premake5.exe vs2022
msbuild build\SPIRV.vcxproj /p:Configuration=Release /p:Platform=x64

clang++ -x hip --offload-arch=amdgcnspirv --cuda-device-only -O2 Test\SPIRV\kernel.hip -o dist\transform.bundle
dist\bin\Release\SPIRV64.exe dist\transform.bundle
```

An optional final argument can select a particular HIP runtime DLL. Record the
driver and DLL versions when testing; this example does not establish a minimum
released Windows driver version.

## Bundle format and integration

Keep the compiler's bundle output enabled. In particular, do not add
`--no-gpu-bundle-output`: that produces raw SPIR-V, which is not the input used by
this module-loading example. The sample checks for an uncompressed offload
bundle before initializing Orochi.

To inspect the generated bundle:

```sh
/opt/rocm/llvm/bin/clang-offload-bundler --list --type=o --input=dist/transform.bundle
```

The device entry should target `amdgcnspirv`; an empty host entry may also be
present. The bundle buffer remains alive until after `oroModuleUnload`.

An application using `OrochiUtils::getFunctionFromPrecompiledBinary` can use the
same bundle format with its existing loading flow. Producing that bundle may
require changes to architecture-specific code in a real kernel. This sample
validates loading and execution of one complete module; it does not demonstrate
linking multiple device modules or support on the CUDA backend.
