//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
#pragma once

#include <cstddef>

enum Api
{
    API_AUTOMATIC,
	API_HIP,
	API_CUDA,
};

enum ppError
{
	ppSuccess = 0,
	ppErrorUnknown = 999,
};

enum ppMemcpyKind
{
    ppMemcpyHostToHost = 0,
    ppMemcpyHostToDevice = 1,
    ppMemcpyDeviceToHost = 2,
    ppMemcpyDeviceToDevice = 3,
    ppMemcpyDefault = 4
};

typedef unsigned int ppU32;
typedef unsigned long long ppDeviceptr;

#ifdef _WIN32
#  define PPAPI __stdcall
#  define PP_CB __stdcall
#else
#  define PPAPI
#  define PP_CB
#endif

typedef int ppDevice;
typedef struct ippCtx_t* ppCtx;
typedef struct ippModule_t* ppModule;
typedef struct ippModuleSymbol_t* ppFunction;
typedef struct ippArray* ppArray;
typedef struct ppMipmappedArray_st* ppMipmappedArray;
typedef struct ippEvent_t* ppEvent;
typedef struct ippStream_t* ppStream;
typedef struct ippPointerAttribute_t* ppPointerAttribute;
typedef unsigned long long ppTextureObject;
typedef void* ppExternalMemory_t;


typedef struct _pprtcProgram* pprtcProgram;

enum pprtcResult
{
	PPRTC_SUCCESS = 0,
	PPRTC_ERROR_INTERNAL_ERROR = 11,
};


typedef enum ppDeviceAttribute {
  ppDeviceAttributeCudaCompatibleBegin = 0,
  ppDeviceAttributeEccEnabled = ppDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
  ppDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
  ppDeviceAttributeAsyncEngineCount,                 ///< Cuda only. Asynchronous engines number.
  ppDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
  ppDeviceAttributeCanUseHostPointerForRegisteredMem,///< Cuda only. Device can access host registered memory
                                                      ///< at the same virtual address as the CPU
  ppDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
  ppDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
  ppDeviceAttributeComputePreemptionSupported,       ///< Cuda only. Device supports Compute Preemption.
  ppDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
  ppDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
  ppDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
  ppDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
  ppDeviceAttributeDeviceOverlap,                    ///< Cuda only. Device can concurrently copy memory and execute a kernel.
                                                      ///< Deprecated. Use instead asyncEngineCount.
  ppDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                      ///< the device without migration
  ppDeviceAttributeGlobalL1CacheSupported,           ///< Cuda only. Device supports caching globals in L1
  ppDeviceAttributeHostNativeAtomicSupported,        ///< Cuda only. Link between the device and the host supports native atomic operations
  ppDeviceAttributeIntegrated,                       ///< Device is integrated GPU
  ppDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
  ppDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
  ppDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
  ppDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
  ppDeviceAttributeLuid,                             ///< Cuda only. 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
  ppDeviceAttributeLuidDeviceNodeMask,               ///< Cuda only. Luid device node mask. Undefined on TCC and non-Windows platforms
  ppDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
  ppDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
  ppDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Cuda only. Max block size per multiprocessor
  ppDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
  ppDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
  ppDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
  ppDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
  ppDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
  ppDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
  ppDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
  ppDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
  ppDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
  ppDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
  ppDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
  ppDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
  ppDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
  ppDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
  ppDeviceAttributeMaxTexture1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered texture.
  ppDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                      ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
  ppDeviceAttributeMaxTexture1DMipmap,               ///< Cuda only. Maximum size of 1D mipmapped texture.
  ppDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
  ppDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
  ppDeviceAttributeMaxTexture2DGather,               ///< Cuda only. Maximum dimensions of 2D texture if gather operations  performed.
  ppDeviceAttributeMaxTexture2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered texture.
  ppDeviceAttributeMaxTexture2DLinear,               ///< Cuda only. Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
  ppDeviceAttributeMaxTexture2DMipmap,               ///< Cuda only. Maximum dimensions of 2D mipmapped texture.
  ppDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
  ppDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
  ppDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
  ppDeviceAttributeMaxTexture3DAlt,                  ///< Cuda only. Maximum dimensions of alternate 3D texture.
  ppDeviceAttributeMaxTextureCubemap,                ///< Cuda only. Maximum dimensions of Cubemap texture
  ppDeviceAttributeMaxTextureCubemapLayered,         ///< Cuda only. Maximum dimensions of Cubemap layered texture.
  ppDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
  ppDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
  ppDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
  ppDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
  ppDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
  ppDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
  ppDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
  ppDeviceAttributeMultiGpuBoardGroupID,             ///< Cuda only. Unique ID of device group on the same multi-GPU board
  ppDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
  ppDeviceAttributeName,                             ///< Device name.
  ppDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                      ///< without calling hipHostRegister on it
  ppDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
  ppDeviceAttributePciBusId,                         ///< PCI Bus ID.
  ppDeviceAttributePciDeviceId,                      ///< PCI Device ID.
  ppDeviceAttributePciDomainID,                      ///< PCI Domain ID.
  ppDeviceAttributePersistingL2CacheMaxSize,         ///< Cuda11 only. Maximum l2 persisting lines capacity in bytes
  ppDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                      ///< by all thread blocks simultaneously resident on a multiprocessor.
  ppDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
  ppDeviceAttributeReservedSharedMemPerBlock,        ///< Cuda11 only. Shared memory reserved by CUDA driver per block.
  ppDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
  ppDeviceAttributeSharedMemPerBlockOptin,           ///< Cuda only. Maximum shared memory per block usable by special opt in.
  ppDeviceAttributeSharedMemPerMultiprocessor,       ///< Cuda only. Shared memory available per multiprocessor.
  ppDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
  ppDeviceAttributeStreamPrioritiesSupported,        ///< Cuda only. Whether to support stream priorities.
  ppDeviceAttributeSurfaceAlignment,                 ///< Cuda only. Alignment requirement for surfaces
  ppDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
  ppDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
  ppDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
  ppDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
  ppDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
  ppDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
  ppDeviceAttributeUuid,                             ///< Cuda only. Unique ID in 16 byte.
  ppDeviceAttributeWarpSize,                         ///< Warp size in threads.
  ppDeviceAttributeCudaCompatibleEnd = 9999,
  ppDeviceAttributeAmdSpecificBegin = 10000,
  ppDeviceAttributeClockInstructionRate = ppDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
  ppDeviceAttributeArch,                                     ///< Device architecture
  ppDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
  ppDeviceAttributeGcnArch,                                  ///< Device gcn architecture
  ppDeviceAttributeGcnArchName,                              ///< Device gcnArch name in 256 bytes
  ppDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
  ppDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
  ppDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                              ///< devices with unmatched functions
  ppDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                              ///< devices with unmatched grid dimensions
  ppDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                              ///< devices with unmatched block dimensions
  ppDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                              ///< devices with unmatched shared memories
  ppDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
  ppDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
  ppDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                              ///< hipStreamWaitValue64() , '0' otherwise.
  ppDeviceAttributeAmdSpecificEnd = 19999,
  ppDeviceAttributeVendorSpecificBegin = 20000,
  // Extended attributes for vendors
} ppDeviceAttribute;

typedef struct PPdevprop_st {
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int sharedMemPerBlock;
    int totalConstantMemory;
    int SIMDWidth;
    int memPitch;
    int regsPerBlock;
    int clockRate;
    int textureAlign;
} PPdevprop;

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

                                     // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

                                         // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

                              // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

                                  // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

                                        // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} ppDeviceArch;

typedef struct ppDeviceProp {
    char name[256];            ///< Device name.
    size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
    int regsPerBlock;          ///< Registers per block.
    int warpSize;              ///< Warp size.
    int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];        ///< Max grid dimensions (XYZ).
    int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;       ///< Max global memory clock frequency in khz.
    int memoryBusWidth;        ///< Global memory bus width in bits.
    size_t totalConstMem;      ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int multiProcessorCount;          ///< Number of multi-processors (compute units).
    int l2CacheSize;                  ///< L2 cache size.
    int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
    int computeMode;                  ///< Compute mode.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    ppDeviceArch arch;      ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;           ///< PCI Domain ID
    int pciBusID;              ///< PCI Bus ID.
    int pciDeviceID;           ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                      ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                     ///< Check whether HIP can map host memory
    int gcnArch;                              ///< DEPRECATED: use gcnArchName instead
    char gcnArchName[256];                    ///< AMD GCN Arch Name.
    int integrated;            ///< APU vs dGPU
    int cooperativeLaunch;            ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch; ///< HIP device supports cooperative launch on multiple devices
    int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
    int maxTexture1D;          ///< Maximum number of elements in 1D images
    int maxTexture2D[2];       ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture3D[3];       ///< Maximum dimensions (width, height, depth) of 3D images, in image elements
    unsigned int* hdpMemFlushCntl;      ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;      ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    size_t memPitch;                 ///<Maximum pitch in bytes allowed by memory copies
    size_t textureAlignment;         ///<Alignment requirement for textures
    size_t texturePitchAlignment;    ///<Pitch alignment requirement for texture references bound to pitched memory
    int kernelExecTimeoutEnabled;    ///<Run time limit for kernels executed on the device
    int ECCEnabled;                  ///<Device has ECC support enabled
    int tccDriver;                   ///< 1:If device is Tesla device using TCC driver, else 0
    int cooperativeMultiDeviceUnmatchedFunc;        ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;     ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;    ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;   ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched shared memories
    int isLargeBar;                  ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                ///< Revision of the GPU in this device
    int managedMemory;               ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost; ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;     ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;        ///< Device supports coherently accessing pageable memory
                                     ///< without calling hipHostRegister on it
    int pageableMemoryAccessUsesHostPageTables; ///< Device accesses pageable memory via the host's page tables
} ppDeviceProp;

typedef enum PPpointer_attribute_enum {
    PP_POINTER_ATTRIBUTE_CONTEXT = 1,
    PP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    PP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    PP_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    PP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    PP_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    PP_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    PP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
} PPpointer_attribute;

typedef enum ppFunction_attribute {
    PP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    PP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    PP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    PP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    PP_FUNC_ATTRIBUTE_NUM_REGS = 4,
    PP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    PP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    PP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    PP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    PP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    PP_FUNC_ATTRIBUTE_MAX,
} ppFunction_attribute;

typedef enum ppFuncCache_t {
    ppFuncCachePreferNone = 0x00,
    ppFuncCachePreferShared = 0x01,
    ppFuncCachePreferL1 = 0x02,
    ppFuncCachePreferEqual = 0x03,
} ppFuncCache_t;

typedef enum ppSharedMemConfig {
    ppSharedMemBankSizeDefault = 0x00,
    ppSharedMemBankSizeFourByte = 0x01,
    ppSharedMemBankSizeEightByte = 0x02,
} ppSharedMemConfig;

typedef enum PPshared_carveout_enum {
    PP_SHAREDMEM_CARVEOUT_DEFAULT,
    PP_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
    PP_SHAREDMEM_CARVEOUT_MAX_L1 = 0,
} PPshared_carveout;



typedef enum ppComputeMode {
    ppComputeModeDefault = 0,
    ppComputeModeProhibited = 2,
    ppComputeModeExclusiveProcess = 3,
} ppComputeMode;

typedef enum PPmem_advise_enum {
    PP_MEM_ADVISE_SET_READ_MOSTLY = 1,
    PP_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    PP_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    PP_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    PP_MEM_ADVISE_SET_ACCESSED_BY = 5,
    PP_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
} PPmem_advise;

typedef enum PPmem_range_attribute_enum {
    PP_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    PP_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    PP_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    PP_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
} PPmem_range_attribute;

typedef enum ppJitOption {
    ppJitOptionMaxRegisters = 0,
    ppJitOptionThreadsPerBlock,
    ppJitOptionWallTime,
    ppJitOptionInfoLogBuffer,
    ppJitOptionInfoLogBufferSizeBytes,
    ppJitOptionErrorLogBuffer,
    ppJitOptionErrorLogBufferSizeBytes,
    ppJitOptionOptimizationLevel,
    ppJitOptionTargetFromContext,
    ppJitOptionTarget,
    ppJitOptionFallbackStrategy,
    ppJitOptionGenerateDebugInfo,
    ppJitOptionLogVerbose,
    ppJitOptionGenerateLineInfo,
    ppJitOptionCacheMode,
    ppJitOptionSm3xOpt,
    ppJitOptionFastCompile,
    ppJitOptionNumOptions,
} ppJitOption;
/*
typedef enum HIPjit_target_enum {
    HIP_TARGET_COMPUTE_20 = 20,
    HIP_TARGET_COMPUTE_21 = 21,
    HIP_TARGET_COMPUTE_30 = 30,
    HIP_TARGET_COMPUTE_32 = 32,
    HIP_TARGET_COMPUTE_35 = 35,
    HIP_TARGET_COMPUTE_37 = 37,
    HIP_TARGET_COMPUTE_50 = 50,
    HIP_TARGET_COMPUTE_52 = 52,
    HIP_TARGET_COMPUTE_53 = 53,
    HIP_TARGET_COMPUTE_60 = 60,
    HIP_TARGET_COMPUTE_61 = 61,
    HIP_TARGET_COMPUTE_62 = 62,
    HIP_TARGET_COMPUTE_70 = 70,
    HIP_TARGET_COMPUTE_73 = 73,
    HIP_TARGET_COMPUTE_75 = 75,
} HIPjit_target;

typedef enum HIPjit_fallback_enum {
    HIP_PREFER_PTX = 0,
    HIP_PREFER_BINARY,
} HIPjit_fallback;

typedef enum HIPjit_cacheMode_enum {
    HIP_JIT_CACHE_OPTION_NONE = 0,
    HIP_JIT_CACHE_OPTION_CG,
    HIP_JIT_CACHE_OPTION_CA,
} HIPjit_cacheMode;

typedef enum HIPjitInputType_enum {
    HIP_JIT_INPUT_HIPBIN = 0,
    HIP_JIT_INPUT_PTX,
    HIP_JIT_INPUT_FATBINARY,
    HIP_JIT_INPUT_OBJECT,
    HIP_JIT_INPUT_LIBRARY,
    HIP_JIT_NUM_INPUT_TYPES,
} HIPjitInputType;

typedef struct HIPlinkState_st* HIPlinkState;

typedef enum hipGLDeviceList {
    hipGLDeviceListAll = 1,           ///< All hip devices used by current OpenGL context.
    hipGLDeviceListCurrentFrame = 2,  ///< Hip devices used by current OpenGL context in current
                                      ///< frame
                                      hipGLDeviceListNextFrame = 3      ///< Hip devices used by current OpenGL context in next
                                                                        ///< frame.
} hipGLDeviceList;

typedef enum hipGraphicsRegisterFlags {
    hipGraphicsRegisterFlagsNone = 0,
    hipGraphicsRegisterFlagsReadOnly = 1,  ///< HIP will not write to this registered resource
    hipGraphicsRegisterFlagsWriteDiscard =
    2,  ///< HIP will only write and will not read from this registered resource
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4,  ///< HIP will bind this resource to a surface
    hipGraphicsRegisterFlagsTextureGather =
    8  ///< HIP will perform texture gather operations on this registered resource
} hipGraphicsRegisterFlags;

typedef enum HIPgraphicsRegisterFlags_enum {
    HIP_GRAPHICS_REGISTER_FLAGS_NONE = 0x00,
    HIP_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 0x01,
    HIP_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02,
    HIP_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04,
    HIP_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08,
} HIPgraphicsRegisterFlags;

typedef enum HIPgraphicsMapResourceFlags_enum {
    HIP_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00,
    HIP_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01,
    HIP_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,
} HIPgraphicsMapResourceFlags;

typedef enum HIParray_cubemap_face_enum {
    HIP_HIPBEMAP_FACE_POSITIVE_X = 0x00,
    HIP_HIPBEMAP_FACE_NEGATIVE_X = 0x01,
    HIP_HIPBEMAP_FACE_POSITIVE_Y = 0x02,
    HIP_HIPBEMAP_FACE_NEGATIVE_Y = 0x03,
    HIP_HIPBEMAP_FACE_POSITIVE_Z = 0x04,
    HIP_HIPBEMAP_FACE_NEGATIVE_Z = 0x05,
} HIParray_cubemap_face;

typedef enum hipLimit_t {
    HIP_LIMIT_STACK_SIZE = 0x00,
    HIP_LIMIT_PRINTF_FIFO_SIZE = 0x01,
    hipLimitMallocHeapSize = 0x02,
    HIP_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
    HIP_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    HIP_LIMIT_MAX,
} hipLimit_t;

typedef enum hipResourceType {
    hipResourceTypeArray = 0x00,
    hipResourceTypeMipmappedArray = 0x01,
    hipResourceTypeLinear = 0x02,
    hipResourceTypePitch2D = 0x03,
} hipResourceType;

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled = 704,
    hipErrorPeerAccessNotEnabled = 705,
    hipErrorSetOnActiveProcess = 708,
    hipErrorAssert = 710,
    hipErrorHostMemoryAlreadyRegistered = 712,
    hipErrorHostMemoryNotRegistered = 713,
    hipErrorLaunchFailure = 719,
    hipErrorCooperativeLaunchTooLarge = 720,
    hipErrorNotSupported = 801,
    hipErrorUnknown = 999,
} hipError_t;
*/

typedef enum ppExternalMemoryHandleType_enum {
  ppExternalMemoryHandleTypeOpaqueFd = 1,
  ppExternalMemoryHandleTypeOpaqueWin32 = 2,
  ppExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
  ppExternalMemoryHandleTypeD3D12Heap = 4,
  ppExternalMemoryHandleTypeD3D12Resource = 5,
  ppExternalMemoryHandleTypeD3D11Resource = 6,
  ppExternalMemoryHandleTypeD3D11ResourceKmt = 7,
} ppExternalMemoryHandleType;
typedef struct ppExternalMemoryHandleDesc_st {
  ppExternalMemoryHandleType type;
  union {
    int fd;
    struct {
      void *handle;
      const void *name;
    } win32;
  } handle;
  unsigned long long size;
  unsigned int flags;
} ppExternalMemoryHandleDesc;
typedef struct ppExternalMemoryBufferDesc_st {
  unsigned long long offset;
  unsigned long long size;
  unsigned int flags;
} ppExternalMemoryBufferDesc;

/**
* Stream CallBack struct
*/

#define __PP_FUNC_DEC( funcName, args ) template<Api API=API_AUTOMATIC> ppError PPAPI funcName args


ppError PPAPI ppGetErrorName(ppError error, const char** pStr);
ppError PPAPI ppGetErrorString(ppError error, const char** pStr);
__PP_FUNC_DEC( ppInit, (unsigned int Flags) );
ppError PPAPI ppDriverGetVersion(int* driverVersion);
ppError PPAPI ppGetDevice(int* device);
ppError PPAPI ppGetDeviceCount(int* count);
ppError PPAPI ppGetDeviceProperties(ppDeviceProp* props, int deviceId);
ppError PPAPI ppDeviceGet(ppDevice* device, int ordinal);
ppError PPAPI ppDeviceGetName(char* name, int len, ppDevice dev);
ppError PPAPI ppDeviceGetAttribute(int* pi, ppDeviceAttribute attrib, ppDevice dev);
ppError PPAPI ppDeviceComputeCapability(int* major, int* minor, ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxRetain(ppCtx* pctx, ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxRelease(ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxSetFlags(ppDevice dev, unsigned int flags);
ppError PPAPI ppDevicePrimaryCtxGetState(ppDevice dev, unsigned int* flags, int* active);
ppError PPAPI ppDevicePrimaryCtxReset(ppDevice dev);
ppError PPAPI ppCtxCreate(ppCtx* pctx, unsigned int flags, ppDevice dev);
ppError PPAPI ppCtxDestroy(ppCtx ctx);
ppError PPAPI ppCtxPushCurrent(ppCtx ctx);
ppError PPAPI ppCtxPopCurrent(ppCtx* pctx);
ppError PPAPI ppCtxSetCurrent(ppCtx ctx);
ppError PPAPI ppCtxGetCurrent(ppCtx* pctx);
ppError PPAPI ppCtxGetDevice(ppDevice* device);
ppError PPAPI ppCtxGetFlags(unsigned int* flags);
ppError PPAPI ppCtxSynchronize(void);
ppError PPAPI ppDeviceSynchronize(void);
//ppError PPAPI ppCtxGetCacheConfig(hipFuncCache_t* pconfig);
//ppError PPAPI ppCtxSetCacheConfig(hipFuncCache_t config);
//ppError PPAPI ppCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
//ppError PPAPI ppCtxSetSharedMemConfig(hipSharedMemConfig config);
ppError PPAPI ppCtxGetApiVersion(ppCtx ctx, unsigned int* version);
ppError PPAPI ppModuleLoad(ppModule* module, const char* fname);
ppError PPAPI ppModuleLoadData(ppModule* module, const void* image);
ppError PPAPI ppModuleLoadDataEx(ppModule* module, const void* image, unsigned int numOptions, ppJitOption* options, void** optionValues);
ppError PPAPI ppModuleUnload(ppModule hmod);
ppError PPAPI ppModuleGetFunction(ppFunction* hfunc, ppModule hmod, const char* name);
ppError PPAPI ppModuleGetGlobal(ppDeviceptr* dptr, size_t* bytes, ppModule hmod, const char* name);
//ppError PPAPI ppModuleGetTexRef(textureReference** pTexRef, ppModule hmod, const char* name);
ppError PPAPI ppMemGetInfo(size_t* free, size_t* total);
ppError PPAPI ppMalloc(ppDeviceptr* dptr, size_t bytesize);
ppError PPAPI ppMalloc2(ppDeviceptr* dptr, size_t bytesize);
ppError PPAPI ppMemAllocPitch(ppDeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
ppError PPAPI ppFree(ppDeviceptr dptr);
ppError PPAPI ppFree2(ppDeviceptr dptr);
//ppError PPAPI ppMemGetAddressRange(ppDeviceptr* pbase, size_t* psize, ppDeviceptr dptr);
//ppError PPAPI ppHostMalloc(void** pp, size_t bytesize, unsigned int flags);
//ppError PPAPI ppHostFree(void* p);
//ppError PPAPI ppMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
//ppError PPAPI ppHostGetDevicePointer(ppDeviceptr* pdptr, void* p, unsigned int Flags);
//ppError PPAPI ppHostGetFlags(unsigned int* pFlags, void* p);
//ppError PPAPI ppMallocManaged(ppDeviceptr* dptr, size_t bytesize, unsigned int flags);
//ppError PPAPI ppDeviceGetByPCIBusId(hipDevice_t* dev, const char* pciBusId);
//ppError PPAPI ppDeviceGetPCIBusId(char* pciBusId, int len, hipDevice_t dev);
//ppError PPAPI ppMemHostUnregister(void* p);
ppError PPAPI ppMemcpy(void *dst, void *src, size_t ByteCount, ppMemcpyKind kind);
//ppError PPAPI ppMemcpyPeer(ppDeviceptr dstDevice, hipCtx_t dstContext, ppDeviceptr srcDevice, hipCtx_t srcContext, size_t ByteCount);
ppError PPAPI ppMemcpyHtoD(ppDeviceptr dstDevice, void* srcHost, size_t ByteCount);
ppError PPAPI ppMemcpyDtoH(void* dstHost, ppDeviceptr srcDevice, size_t ByteCount);
ppError PPAPI ppMemcpyDtoD(ppDeviceptr dstDevice, ppDeviceptr srcDevice, size_t ByteCount);
//ppError PPAPI ppDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
//ppError PPAPI ppMemcpyParam2D(const hip_Memcpy2D* pCopy);
//ppError PPAPI ppDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
//ppError PPAPI ppMemcpyHtoDAsync(ppDeviceptr dstDevice, const void* srcHost, size_t ByteCount, hipStream_t hStream);
//ppError PPAPI ppMemcpyDtoHAsync(void* dstHost, ppDeviceptr srcDevice, size_t ByteCount, hipStream_t hStream);
//ppError PPAPI ppMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t hStream);
//ppError PPAPI ppDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t hStream);
ppError PPAPI ppMemset(ppDeviceptr dstDevice, unsigned int ui, size_t N);
ppError PPAPI ppMemsetD8(ppDeviceptr dstDevice, unsigned char ui, size_t N);
ppError PPAPI ppMemsetD16(ppDeviceptr dstDevice, unsigned short ui, size_t N);
ppError PPAPI ppMemsetD32(ppDeviceptr dstDevice, unsigned int ui, size_t N);
//ppError PPAPI ppMemsetD8Async(ppDeviceptr dstDevice, unsigned char uc, size_t N, ppStream hStream);
//ppError PPAPI ppMemsetD16Async(ppDeviceptr dstDevice, unsigned short us, size_t N, ppStream hStream);
//ppError PPAPI ppMemsetD32Async(ppDeviceptr dstDevice, unsigned int ui, size_t N, ppStream hStream);
//ppError PPAPI ppMemsetD2D8Async(ppDeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, ppStream hStream);
//ppError PPAPI ppMemsetD2D16Async(ppDeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, ppStream hStream);
//ppError PPAPI ppMemsetD2D32Async(ppDeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, ppStream hStream);
//ppError PPAPI ppArrayCreate(hArray ** pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
//ppError PPAPI ppArrayDestroy(hArray hArray);
//ppError PPAPI ppArray3DCreate(hArray * pHandle, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
ppError PPAPI ppPointerGetAttributes(ppPointerAttribute* attr, ppDeviceptr dptr);
ppError PPAPI ppStreamCreate(ppStream* stream);
//ppError PPAPI ppStreamCreateWithFlags(ppStream* phStream, unsigned int Flags);
//ppError PPAPI ppStreamCreateWithPriority(ppStream* phStream, unsigned int flags, int priority);
//ppError PPAPI ppStreamGetPriority(ppStream hStream, int* priority);
//ppError PPAPI ppStreamGetFlags(ppStream hStream, unsigned int* flags);
//ppError PPAPI ppStreamWaitEvent(ppStream hStream, hipEvent_t hEvent, unsigned int Flags);
//ppError PPAPI ppStreamAddCallback(ppStream hStream, hipStreamCallback_t callback, void* userData, unsigned int flags);
//ppError PPAPI ppStreamQuery(ppStream hStream);
//ppError PPAPI ppStreamSynchronize(ppStream hStream);
//ppError PPAPI ppStreamDestroy(ppStream hStream);
//ppError PPAPI ppEventCreateWithFlags(hipEvent_t* phEvent, unsigned int Flags);
//ppError PPAPI ppEventRecord(hipEvent_t hEvent, ppStream hStream);
//ppError PPAPI ppEventQuery(hipEvent_t hEvent);
//ppError PPAPI ppEventSynchronize(hipEvent_t hEvent);
//ppError PPAPI ppEventDestroy(hipEvent_t hEvent);
//ppError PPAPI ppEventElapsedTime(float* pMilliseconds, hipEvent_t hStart, hipEvent_t hEnd);
//ppError PPAPI ppFuncGetAttribute(int* pi, hipFunction_attribute attrib, hipFunction_t hfunc);
//ppError PPAPI ppFuncSetCacheConfig(hipFunction_t hfunc, hipFuncCache_t config);
ppError PPAPI ppModuleLaunchKernel(ppFunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, ppStream hStream, void** kernelParams, void** extra);
//ppError PPAPI ppDrvOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t func, int blockSize, size_t dynamicSMemSize);
//ppError PPAPI ppDrvOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, hipFunction_t func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
//ppError PPAPI ppModuleOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, hipFunction_t func, size_t dynamicSMemSize, int blockSizeLimit);
//ppError PPAPI ppTexRefSetArray(hipTexRef hTexRef, hArray * hArray, unsigned int Flags);
//ppError PPAPI ppTexRefSetAddress(size_t* ByteOffset, hipTexRef hTexRef, hipDeviceptr_t dptr, size_t bytes);
//ppError PPAPI ppTexRefSetAddress2D(hipTexRef hTexRef, const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr, size_t Pitch);
//ppError PPAPI ppTexRefSetFormat(hipTexRef hTexRef, hipArray_Format fmt, int NumPackedComponents);
//ppError PPAPI ppTexRefSetAddressMode(hipTexRef hTexRef, int dim, hipTextureAddressMode am);
//ppError PPAPI ppTexRefSetFilterMode(hipTexRef hTexRef, hipTextureFilterMode fm);
//ppError PPAPI ppTexRefSetFlags(hipTexRef hTexRef, unsigned int Flags);
//ppError PPAPI ppTexRefGetAddress(hipDeviceptr_t* pdptr, hipTexRef hTexRef);
//ppError PPAPI ppTexRefGetArray(hArray ** phArray, hipTexRef hTexRef);
//ppError PPAPI ppTexRefGetAddressMode(hipTextureAddressMode* pam, hipTexRef hTexRef, int dim);
//ppError PPAPI ppTexObjectCreate(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc, const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
//ppError PPAPI ppTexObjectDestroy(hipTextureObject_t texObject);
//ppError PPAPI ppDeviceCanAccessPeer(int* canAccessPeer, hipDevice_t dev, hipDevice_t peerDev);
//ppError PPAPI ppCtxEnablePeerAccess(hipCtx_t peerContext, unsigned int Flags);
//ppError PPAPI ppCtxDisablePeerAccess(hipCtx_t peerContext);
//ppError PPAPI ppDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attrib, hipDevice_t srcDevice, hipDevice_t dstDevice);
//ppError PPAPI ppGraphicsUnregisterResource(hipGraphicsResource resource);
//ppError PPAPI ppGraphicsResourceGetMappedMipmappedArray(hipMipmappedArray_t* pMipmappedArray, hipGraphicsResource resource);
//ppError PPAPI ppGraphicsResourceGetMappedPointer(hipDeviceptr_t* pDevPtr, size_t* pSize, hipGraphicsResource resource);
//ppError PPAPI ppGraphicsMapResources(unsigned int count, hipGraphicsResource* resources, ppStream hStream);
//ppError PPAPI ppGraphicsUnmapResources(unsigned int count, hipGraphicsResource* resources, ppStream hStream);
//ppError PPAPI ppGraphicsGLRegisterBuffer(hipGraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags);
//ppError PPAPI ppGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList);
ppError PPAPI ppImportExternalMemory(ppExternalMemory_t* extMem_out, const ppExternalMemoryHandleDesc* memHandleDesc);
ppError PPAPI ppExternalMemoryGetMappedBuffer(void **devPtr, ppExternalMemory_t extMem, const ppExternalMemoryBufferDesc* bufferDesc);
ppError PPAPI ppDestroyExternalMemory(ppExternalMemory_t extMem);
ppError PPAPI ppGetLastError(ppError pp_error);
pprtcResult PPAPI pprtcGetErrorString(pprtcResult result);
pprtcResult PPAPI pprtcAddNameExpression(pprtcProgram prog, const char* name_expression);
pprtcResult PPAPI pprtcCompileProgram(pprtcProgram prog, int numOptions, const char** options);
pprtcResult PPAPI pprtcCreateProgram(pprtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames);
pprtcResult PPAPI pprtcDestroyProgram(pprtcProgram* prog);
pprtcResult PPAPI pprtcGetLoweredName(pprtcProgram prog, const char* name_expression, const char** lowered_name);
pprtcResult PPAPI pprtcGetProgramLog(pprtcProgram prog, char* log);
pprtcResult PPAPI pprtcGetProgramLogSize(pprtcProgram prog, size_t* logSizeRet);
pprtcResult PPAPI pprtcGetCode(pprtcProgram prog, char* code);
pprtcResult PPAPI pprtcGetCodeSize(pprtcProgram prog, size_t* codeSizeRet);


enum {
	PP_SUCCESS = 0,
	PP_ERROR_OPEN_FAILED = -1,
	PP_ERROR_ATEXIT_FAILED = -2,
	PP_ERROR_OLD_DRIVER = -3,
};


int ppInitialize( Api api, ppU32 flags );
Api ppGetCurAPI( ppU32 flags );


#include <stdint.h>

//typedef struct dim3 {
//    uint32_t x;  ///< x
//    uint32_t y;  ///< y
//    uint32_t z;  ///< z
//#ifdef __cplusplus
//    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
//#endif
//} dim3;

