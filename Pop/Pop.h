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

enum Api
{
	API_HIP,
	API_CUDA,
};

enum ppError
{
	ppSuccess = 0,
	ppErrorUnknown = 999,
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
typedef int ppDeviceAttribute;
typedef unsigned long long ppTextureObject;


typedef struct _pprtcProgram* pprtcProgram;

enum pprtcResult
{
	PPRTC_SUCCESS = 0,
	PPRTC_ERROR_INTERNAL_ERROR = 11,
};

typedef enum ppJitOption {
/*    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionWallTime,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionSm3xOpt,
    hipJitOptionFastCompile,
    hipJitOptionNumOptions,
*/
} ppJitOption;

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

ppError PPAPI ppGetErrorName(ppError error, const char** pStr);
ppError PPAPI ppGetErrorString(ppError error, const char** pStr);
ppError PPAPI ppInit(unsigned int Flags);
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
ppError PPAPI ppMemAllocPitch(ppDeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
ppError PPAPI ppFree(ppDeviceptr dptr);


//----
ppError PPAPI ppMemcpyHtoD(ppDeviceptr dstDevice, void* srcHost, size_t ByteCount);
ppError PPAPI ppMemcpyDtoH(void* dstHost, ppDeviceptr srcDevice, size_t ByteCount);
ppError PPAPI ppMemcpyDtoD(ppDeviceptr dstDevice, ppDeviceptr srcDevice, size_t ByteCount);
ppError PPAPI ppMemset(ppDeviceptr dstDevice, unsigned int ui, size_t N);
ppError PPAPI ppMemsetD8(ppDeviceptr dstDevice, unsigned char ui, size_t N);
ppError PPAPI ppMemsetD16(ppDeviceptr dstDevice, unsigned short ui, size_t N);
ppError PPAPI ppMemsetD32(ppDeviceptr dstDevice, unsigned int ui, size_t N);


//----
ppError PPAPI ppModuleLaunchKernel(ppFunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, ppStream hStream, void** kernelParams, void** extra);

//----
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

//----
ppError PPAPI ppPointerGetAttributes(ppPointerAttribute* attr, ppDeviceptr dptr);

//----
ppError PPAPI ppStreamCreate(ppStream* stream);


enum {
	PP_SUCCESS = 0,
	PP_ERROR_OPEN_FAILED = -1,
	PP_ERROR_ATEXIT_FAILED = -2,
	PP_ERROR_OLD_DRIVER = -3,
};


int ppInitialize( Api api, ppU32 flags );


#include <stdint.h>

typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
#ifdef __cplusplus
    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
} dim3;
