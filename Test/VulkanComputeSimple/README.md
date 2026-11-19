# VulkanComputeSimple - Simple Vulkan Oro Interop

## Description

A Vulkan compute shader fills a buffer allocated from exportable device memory. That buffer is then imported on the Orochi side and read back to verify its contents. The Vulkan physical device is matched to the Orochi device by PCI domain/bus/device, so both APIs act on the same GPU. Memory is shared through `VK_KHR_external_memory_win32` on Windows and `VK_KHR_external_memory_fd` on Linux. Synchronization is a plain `vkDeviceWaitIdle` rather than a shared semaphore, so the sharing is one way: Vulkan writes, Orochi reads.

## Involved functions

oroImportExternalMemory, oroExternalMemoryGetMappedBuffer, oroDestroyExternalMemory

## Build and Run

The demo is built as part of the normal Orochi build and runs as the `VulkanComputeSimple64` unit test case.

The Vulkan headers are the only extra build dependency. No Vulkan library is linked, because vulkan-hpp's RAII `Context` loads the loader (`vulkan-1.dll` / `libvulkan.so.1`) dynamically at runtime.

Windows uses the headers bundled in `vulkan/` and `vk_video/`. Those are too old to compile against a modern libstdc++, so Linux uses either a LunarG SDK (when `VULKAN_SDK` is set, for example by sourcing the SDK's `setup-env.sh`) or the system headers from `libvulkan-dev`. If neither is found, the project is skipped and the unit test case is not registered.

```sh
# Linux
sudo apt install libvulkan-dev        # or: source ~/vulkan/<version>/setup-env.sh
./tools/premake5/linux64/premake5 gmake
make config=release_x64
```

The demo loads its SPIR-V from `../Test/VulkanComputeSimple/main.comp.spv`, relative to the working directory, so it expects to run from a directory one level below the repository root, which is where the unit test runs it from.

To build it standalone instead:

```
clang main.cpp ../../Orochi/Orochi.cpp ../../contrib/cuew/src/cuew.cpp ../../contrib/hipew/src/hipew.cpp -I./ -I../../ -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -lversion -std=c++17
```

```sh
# Linux
g++ main.cpp ../../Orochi/Orochi.cpp ../../contrib/cuew/src/cuew.cpp ../../contrib/hipew/src/hipew.cpp -I../../ -ldl -std=c++17 -o VulkanComputeSimple64
```

## Regenerating the shader

`main.comp.spv` is checked in. To rebuild it after editing `main.comp`:

```
clang-format -i main.cpp main.comp ; glslangValidator --target-env vulkan1.2 -V main.comp -o main.comp.spv ; spirv-opt -O main.comp.spv -o main.comp.spv
```
