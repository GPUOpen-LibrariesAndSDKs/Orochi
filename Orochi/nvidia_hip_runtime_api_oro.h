/*
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>


#define CUDA_9000 9000
#define CUDA_10010 10010
#define CUDA_10020 10020
#define CUDA_11010 11010
#define CUDA_11020 11020
#define CUDA_11030 11030
#define CUDA_11040 11040
#define CUDA_11060 11060
#define CUDA_12000 12000

#ifdef __cplusplus
extern "C" {
#endif














enum hipError_t
{
	hipSuccess = 0,
	hipErrorInvalidValue = 1,
	hipErrorOutOfMemory = 2,
	hipErrorMemoryAllocation = 2,
	hipErrorNotInitialized = 3,
	hipErrorInitializationError = 3,
	hipErrorDeinitialized = 4,
	hipErrorProfilerDisabled = 5,
	hipErrorProfilerNotInitialized = 6,
	hipErrorProfilerAlreadyStarted = 7,
	hipErrorProfilerAlreadyStopped = 8,
	hipErrorInvalidConfiguration = 9,
	hipErrorInvalidPitchValue = 12,
	hipErrorInvalidSymbol = 13,
	hipErrorInvalidDevicePointer = 17,
	hipErrorInvalidMemcpyDirection = 21,
	hipErrorInsufficientDriver = 35,
	hipErrorMissingConfiguration = 52,
	hipErrorPriorLaunchFailure = 53,
	hipErrorInvalidDeviceFunction = 98,
	hipErrorNoDevice = 100,
	hipErrorInvalidDevice = 101,
	hipErrorInvalidImage = 200,
	hipErrorInvalidContext = 201,
	hipErrorContextAlreadyCurrent = 202,
	hipErrorMapFailed = 205,
	hipErrorMapBufferObjectFailed = 205,
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
	hipErrorInvalidResourceHandle = 400,
	hipErrorIllegalState = 401,
	hipErrorNotFound = 500,
	hipErrorNotReady = 600,
	hipErrorIllegalAddress = 700,
	hipErrorLaunchOutOfResources = 701,
	hipErrorLaunchTimeOut = 702,
	hipErrorPeerAccessAlreadyEnabled = 704,
	hipErrorPeerAccessNotEnabled = 705,
	hipErrorSetOnActiveProcess = 708,
	hipErrorContextIsDestroyed = 709,
	hipErrorAssert = 710,
	hipErrorHostMemoryAlreadyRegistered = 712,
	hipErrorHostMemoryNotRegistered = 713,
	hipErrorLaunchFailure = 719,
	hipErrorCooperativeLaunchTooLarge = 720,
	hipErrorNotSupported = 801,
	hipErrorStreamCaptureUnsupported = 900,
	hipErrorStreamCaptureInvalidated = 901,
	hipErrorStreamCaptureMerge = 902,
	hipErrorStreamCaptureUnmatched = 903,
	hipErrorStreamCaptureUnjoined = 904,
	hipErrorStreamCaptureIsolation = 905,
	hipErrorStreamCaptureImplicit = 906,
	hipErrorCapturedEvent = 907,
	hipErrorStreamCaptureWrongThread = 908,
	hipErrorGraphExecUpdateFailure = 910,
	hipErrorUnknown = 999,
	hipErrorRuntimeMemory = 1052,
	hipErrorRuntimeOther = 1053,
	hipErrorTbd = 1054,
};
typedef enum hipError_t hipError_t;
struct hipDeviceArch_t
{
	unsigned int hasGlobalInt32Atomics;
	unsigned int hasGlobalFloatAtomicExch;
	unsigned int hasSharedInt32Atomics;
	unsigned int hasSharedFloatAtomicExch;
	unsigned int hasFloatAtomicAdd;
	unsigned int hasGlobalInt64Atomics;
	unsigned int hasSharedInt64Atomics;
	unsigned int hasDoubles;
	unsigned int hasWarpVote;
	unsigned int hasWarpBallot;
	unsigned int hasWarpShuffle;
	unsigned int hasFunnelShift;
	unsigned int hasThreadFenceSystem;
	unsigned int hasSyncThreadsExt;
	unsigned int hasSurfaceFuncs;
	unsigned int has3dGrid;
	unsigned int hasDynamicParallelism;
};
typedef struct hipDeviceArch_t hipDeviceArch_t;
struct hipUUID_t
{
	char bytes[16];
};
typedef struct hipUUID_t hipUUID;
struct hipDeviceProp_t
{
	char name[256];
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	int clockRate;
	int memoryClockRate;
	int memoryBusWidth;
	size_t totalConstMem;
	int major;
	int minor;
	int multiProcessorCount;
	int l2CacheSize;
	int maxThreadsPerMultiProcessor;
	int computeMode;
	int clockInstructionRate;
	hipDeviceArch_t arch;
	int concurrentKernels;
	int pciDomainID;
	int pciBusID;
	int pciDeviceID;
	size_t maxSharedMemoryPerMultiProcessor;
	int isMultiGpuBoard;
	int canMapHostMemory;
	int gcnArch;
	char gcnArchName[256];
	int integrated;
	int cooperativeLaunch;
	int cooperativeMultiDeviceLaunch;
	int maxTexture1DLinear;
	int maxTexture1D;
	int maxTexture2D[2];
	int maxTexture3D[3];
	unsigned int * hdpMemFlushCntl;
	unsigned int * hdpRegFlushCntl;
	size_t memPitch;
	size_t textureAlignment;
	size_t texturePitchAlignment;
	int kernelExecTimeoutEnabled;
	int ECCEnabled;
	int tccDriver;
	int cooperativeMultiDeviceUnmatchedFunc;
	int cooperativeMultiDeviceUnmatchedGridDim;
	int cooperativeMultiDeviceUnmatchedBlockDim;
	int cooperativeMultiDeviceUnmatchedSharedMem;
	int isLargeBar;
	int asicRevision;
	int managedMemory;
	int directManagedMemAccessFromHost;
	int concurrentManagedAccess;
	int pageableMemoryAccess;
	int pageableMemoryAccessUsesHostPageTables;
};
typedef struct hipDeviceProp_t hipDeviceProp_t;
enum hipDeviceAttribute_t
{
	hipDeviceAttributeCudaCompatibleBegin = 0,
	hipDeviceAttributeEccEnabled = 0,
	hipDeviceAttributeAccessPolicyMaxWindowSize = 1,
	hipDeviceAttributeAsyncEngineCount = 2,
	hipDeviceAttributeCanMapHostMemory = 3,
	hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4,
	hipDeviceAttributeClockRate = 5,
	hipDeviceAttributeComputeMode = 6,
	hipDeviceAttributeComputePreemptionSupported = 7,
	hipDeviceAttributeConcurrentKernels = 8,
	hipDeviceAttributeConcurrentManagedAccess = 9,
	hipDeviceAttributeCooperativeLaunch = 10,
	hipDeviceAttributeCooperativeMultiDeviceLaunch = 11,
	hipDeviceAttributeDeviceOverlap = 12,
	hipDeviceAttributeDirectManagedMemAccessFromHost = 13,
	hipDeviceAttributeGlobalL1CacheSupported = 14,
	hipDeviceAttributeHostNativeAtomicSupported = 15,
	hipDeviceAttributeIntegrated = 16,
	hipDeviceAttributeIsMultiGpuBoard = 17,
	hipDeviceAttributeKernelExecTimeout = 18,
	hipDeviceAttributeL2CacheSize = 19,
	hipDeviceAttributeLocalL1CacheSupported = 20,
	hipDeviceAttributeLuid = 21,
	hipDeviceAttributeLuidDeviceNodeMask = 22,
	hipDeviceAttributeComputeCapabilityMajor = 23,
	hipDeviceAttributeManagedMemory = 24,
	hipDeviceAttributeMaxBlocksPerMultiProcessor = 25,
	hipDeviceAttributeMaxBlockDimX = 26,
	hipDeviceAttributeMaxBlockDimY = 27,
	hipDeviceAttributeMaxBlockDimZ = 28,
	hipDeviceAttributeMaxGridDimX = 29,
	hipDeviceAttributeMaxGridDimY = 30,
	hipDeviceAttributeMaxGridDimZ = 31,
	hipDeviceAttributeMaxSurface1D = 32,
	hipDeviceAttributeMaxSurface1DLayered = 33,
	hipDeviceAttributeMaxSurface2D = 34,
	hipDeviceAttributeMaxSurface2DLayered = 35,
	hipDeviceAttributeMaxSurface3D = 36,
	hipDeviceAttributeMaxSurfaceCubemap = 37,
	hipDeviceAttributeMaxSurfaceCubemapLayered = 38,
	hipDeviceAttributeMaxTexture1DWidth = 39,
	hipDeviceAttributeMaxTexture1DLayered = 40,
	hipDeviceAttributeMaxTexture1DLinear = 41,
	hipDeviceAttributeMaxTexture1DMipmap = 42,
	hipDeviceAttributeMaxTexture2DWidth = 43,
	hipDeviceAttributeMaxTexture2DHeight = 44,
	hipDeviceAttributeMaxTexture2DGather = 45,
	hipDeviceAttributeMaxTexture2DLayered = 46,
	hipDeviceAttributeMaxTexture2DLinear = 47,
	hipDeviceAttributeMaxTexture2DMipmap = 48,
	hipDeviceAttributeMaxTexture3DWidth = 49,
	hipDeviceAttributeMaxTexture3DHeight = 50,
	hipDeviceAttributeMaxTexture3DDepth = 51,
	hipDeviceAttributeMaxTexture3DAlt = 52,
	hipDeviceAttributeMaxTextureCubemap = 53,
	hipDeviceAttributeMaxTextureCubemapLayered = 54,
	hipDeviceAttributeMaxThreadsDim = 55,
	hipDeviceAttributeMaxThreadsPerBlock = 56,
	hipDeviceAttributeMaxThreadsPerMultiProcessor = 57,
	hipDeviceAttributeMaxPitch = 58,
	hipDeviceAttributeMemoryBusWidth = 59,
	hipDeviceAttributeMemoryClockRate = 60,
	hipDeviceAttributeComputeCapabilityMinor = 61,
	hipDeviceAttributeMultiGpuBoardGroupID = 62,
	hipDeviceAttributeMultiprocessorCount = 63,
	hipDeviceAttributeName = 64,
	hipDeviceAttributePageableMemoryAccess = 65,
	hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66,
	hipDeviceAttributePciBusId = 67,
	hipDeviceAttributePciDeviceId = 68,
	hipDeviceAttributePciDomainID = 69,
	hipDeviceAttributePersistingL2CacheMaxSize = 70,
	hipDeviceAttributeMaxRegistersPerBlock = 71,
	hipDeviceAttributeMaxRegistersPerMultiprocessor = 72,
	hipDeviceAttributeReservedSharedMemPerBlock = 73,
	hipDeviceAttributeMaxSharedMemoryPerBlock = 74,
	hipDeviceAttributeSharedMemPerBlockOptin = 75,
	hipDeviceAttributeSharedMemPerMultiprocessor = 76,
	hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77,
	hipDeviceAttributeStreamPrioritiesSupported = 78,
	hipDeviceAttributeSurfaceAlignment = 79,
	hipDeviceAttributeTccDriver = 80,
	hipDeviceAttributeTextureAlignment = 81,
	hipDeviceAttributeTexturePitchAlignment = 82,
	hipDeviceAttributeTotalConstantMemory = 83,
	hipDeviceAttributeTotalGlobalMem = 84,
	hipDeviceAttributeUnifiedAddressing = 85,
	hipDeviceAttributeUuid = 86,
	hipDeviceAttributeWarpSize = 87,
	hipDeviceAttributeMemoryPoolsSupported = 88,
	hipDeviceAttributeVirtualMemoryManagementSupported = 89,
	hipDeviceAttributeCudaCompatibleEnd = 9999,
	hipDeviceAttributeAmdSpecificBegin = 10000,
	hipDeviceAttributeClockInstructionRate = 10000,
	hipDeviceAttributeArch = 10001,
	hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002,
	hipDeviceAttributeGcnArch = 10003,
	hipDeviceAttributeGcnArchName = 10004,
	hipDeviceAttributeHdpMemFlushCntl = 10005,
	hipDeviceAttributeHdpRegFlushCntl = 10006,
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007,
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008,
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009,
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010,
	hipDeviceAttributeIsLargeBar = 10011,
	hipDeviceAttributeAsicRevision = 10012,
	hipDeviceAttributeCanUseStreamWaitValue = 10013,
	hipDeviceAttributeImageSupport = 10014,
	hipDeviceAttributePhysicalMultiProcessorCount = 10015,
	hipDeviceAttributeFineGrainSupport = 10016,
	hipDeviceAttributeWallClockRate = 10017,
	hipDeviceAttributeAmdSpecificEnd = 19999,
	hipDeviceAttributeVendorSpecificBegin = 20000,
};
typedef enum hipDeviceAttribute_t hipDeviceAttribute_t;
enum hipMemoryType
{
	hipMemoryTypeHost = 0,
	hipMemoryTypeDevice = 1,
	hipMemoryTypeArray = 2,
	hipMemoryTypeUnified = 3,
	hipMemoryTypeManaged = 4,
};
typedef enum hipMemoryType hipMemoryType;
struct hipPointerAttribute_t
{
	union 
	{
		enum hipMemoryType memoryType;
		enum hipMemoryType type;
	};
	int device;
	void * devicePointer;
	void * hostPointer;
	int isManaged;
	unsigned int allocationFlags;
};
typedef struct hipPointerAttribute_t hipPointerAttribute_t;

















#ifdef __cplusplus
#define __dparm(x) = x
#else
#define __dparm(x)
#endif

// Add Deprecated Support for CUDA Mapped HIP APIs
#if defined(__DOXYGEN_ONLY__) || defined(HIP_ENABLE_DEPRECATED)
#define __HIP_DEPRECATED
#elif defined(_MSC_VER)
#define __HIP_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __HIP_DEPRECATED __attribute__((deprecated))
#else
#define __HIP_DEPRECATED
#endif

// Add Deprecated Support for CUDA Mapped HIP APIs
#if defined(__DOXYGEN_ONLY__) || defined(HIP_ENABLE_DEPRECATED)
#define __HIP_DEPRECATED_MSG(msg)
#elif defined(_MSC_VER)
#define __HIP_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__)
#define __HIP_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#else
#define __HIP_DEPRECATED_MSG(msg)
#endif


// TODO -move to include/hip_runtime_api.h as a common implementation.
/**
 * Memory copy types
 *
 */
typedef enum cudaMemcpyKind hipMemcpyKind;
#define hipMemcpyHostToHost cudaMemcpyHostToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyDefault cudaMemcpyDefault

typedef enum hipMemoryAdvise {
    hipMemAdviseSetReadMostly,
    hipMemAdviseUnsetReadMostly,
    hipMemAdviseSetPreferredLocation,
    hipMemAdviseUnsetPreferredLocation,
    hipMemAdviseSetAccessedBy,
    hipMemAdviseUnsetAccessedBy
} hipMemoryAdvise;

// hipDataType
#define hipDataType cudaDataType
#define HIP_R_16F  CUDA_R_16F
#define HIP_C_16F  CUDA_C_16F
#define HIP_R_16BF CUDA_R_16BF
#define HIP_C_16BF CUDA_C_16BF
#define HIP_R_32F  CUDA_R_32F
#define HIP_C_32F  CUDA_C_32F
#define HIP_R_64F  CUDA_R_64F
#define HIP_C_64F  CUDA_C_64F
#define HIP_R_4I   CUDA_R_4I
#define HIP_C_4I   CUDA_C_4I
#define HIP_R_4U   CUDA_R_4U
#define HIP_C_4U   CUDA_C_4U
#define HIP_R_8I   CUDA_R_8I
#define HIP_C_8I   CUDA_C_8I
#define HIP_R_8U   CUDA_R_8U
#define HIP_C_8U   CUDA_C_8U
#define HIP_R_16I  CUDA_R_16I
#define HIP_C_16I  CUDA_C_16I
#define HIP_R_16U  CUDA_R_16U
#define HIP_C_16U  CUDA_C_16U
#define HIP_R_32I  CUDA_R_32I
#define HIP_C_32I  CUDA_C_32I
#define HIP_R_32U  CUDA_R_32U
#define HIP_C_32U  CUDA_C_32U
#define HIP_R_64I  CUDA_R_64I
#define HIP_C_64I  CUDA_C_64I
#define HIP_R_64U  CUDA_R_64U
#define HIP_C_64U  CUDA_C_64U

// hip stream operation masks
#define STREAM_OPS_WAIT_MASK_32 0xFFFFFFFF
#define STREAM_OPS_WAIT_MASK_64 0xFFFFFFFFFFFFFFFF

// stream operation flags
#define hipStreamWaitValueGte CU_STREAM_WAIT_VALUE_GEQ
#define hipStreamWaitValueEq  CU_STREAM_WAIT_VALUE_EQ
#define hipStreamWaitValueAnd CU_STREAM_WAIT_VALUE_AND
#define hipStreamWaitValueNor CU_STREAM_WAIT_VALUE_NOR

// hipLibraryPropertyType
#define hipLibraryPropertyType libraryPropertyType
#define HIP_LIBRARY_MAJOR_VERSION MAJOR_VERSION
#define HIP_LIBRARY_MINOR_VERSION MINOR_VERSION
#define HIP_LIBRARY_PATCH_LEVEL PATCH_LEVEL

#define HIP_ARRAY_DESCRIPTOR CUDA_ARRAY_DESCRIPTOR
#define HIP_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR

//hipArray_Format
#define HIP_AD_FORMAT_UNSIGNED_INT8   CU_AD_FORMAT_UNSIGNED_INT8
#define HIP_AD_FORMAT_UNSIGNED_INT16  CU_AD_FORMAT_UNSIGNED_INT16
#define HIP_AD_FORMAT_UNSIGNED_INT32  CU_AD_FORMAT_UNSIGNED_INT32
#define HIP_AD_FORMAT_SIGNED_INT8     CU_AD_FORMAT_SIGNED_INT8
#define HIP_AD_FORMAT_SIGNED_INT16    CU_AD_FORMAT_SIGNED_INT16
#define HIP_AD_FORMAT_SIGNED_INT32    CU_AD_FORMAT_SIGNED_INT32
#define HIP_AD_FORMAT_HALF            CU_AD_FORMAT_HALF
#define HIP_AD_FORMAT_FLOAT           CU_AD_FORMAT_FLOAT

// hipArray_Format
#define hipArray_Format CUarray_format

inline static CUarray_format hipArray_FormatToCUarray_format(
    hipArray_Format format) {
    switch (format) {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return CU_AD_FORMAT_UNSIGNED_INT8;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return CU_AD_FORMAT_UNSIGNED_INT16;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        case HIP_AD_FORMAT_SIGNED_INT8:
            return CU_AD_FORMAT_SIGNED_INT8;
        case HIP_AD_FORMAT_SIGNED_INT16:
            return CU_AD_FORMAT_SIGNED_INT16;
        case HIP_AD_FORMAT_SIGNED_INT32:
            return CU_AD_FORMAT_SIGNED_INT32;
        case HIP_AD_FORMAT_HALF:
            return CU_AD_FORMAT_HALF;
        case HIP_AD_FORMAT_FLOAT:
            return CU_AD_FORMAT_FLOAT;
        default:
            return CU_AD_FORMAT_UNSIGNED_INT8;
    }
}

#define HIP_TR_ADDRESS_MODE_WRAP   CU_TR_ADDRESS_MODE_WRAP
#define HIP_TR_ADDRESS_MODE_CLAMP  CU_TR_ADDRESS_MODE_CLAMP
#define HIP_TR_ADDRESS_MODE_MIRROR CU_TR_ADDRESS_MODE_MIRROR
#define HIP_TR_ADDRESS_MODE_BORDER CU_TR_ADDRESS_MODE_BORDER

// hipAddress_mode
#define hipAddress_mode CUaddress_mode

inline static CUaddress_mode hipAddress_modeToCUaddress_mode(
    hipAddress_mode mode) {
    switch (mode) {
        case HIP_TR_ADDRESS_MODE_WRAP:
            return CU_TR_ADDRESS_MODE_WRAP;
        case HIP_TR_ADDRESS_MODE_CLAMP:
            return CU_TR_ADDRESS_MODE_CLAMP;
        case HIP_TR_ADDRESS_MODE_MIRROR:
            return CU_TR_ADDRESS_MODE_MIRROR;
        case HIP_TR_ADDRESS_MODE_BORDER:
            return CU_TR_ADDRESS_MODE_BORDER;
        default:
            return CU_TR_ADDRESS_MODE_WRAP;
    }
}

#define HIP_TR_FILTER_MODE_POINT   CU_TR_FILTER_MODE_POINT
#define HIP_TR_FILTER_MODE_LINEAR  CU_TR_FILTER_MODE_LINEAR

// hipFilter_mode
#define hipFilter_mode CUfilter_mode

inline static CUfilter_mode hipFilter_mode_enumToCUfilter_mode(
    hipFilter_mode mode) {
    switch (mode) {
        case HIP_TR_FILTER_MODE_POINT:
            return CU_TR_FILTER_MODE_POINT;
        case HIP_TR_FILTER_MODE_LINEAR:
            return CU_TR_FILTER_MODE_LINEAR;
        default:
            return CU_TR_FILTER_MODE_POINT;
    }
}

//hipResourcetype
#define HIP_RESOURCE_TYPE_ARRAY            CU_RESOURCE_TYPE_ARRAY
#define HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY  CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define HIP_RESOURCE_TYPE_LINEAR           CU_RESOURCE_TYPE_LINEAR
#define HIP_RESOURCE_TYPE_PITCH2D          CU_RESOURCE_TYPE_PITCH2D

// hipResourcetype
#define hipResourcetype CUresourcetype

inline static CUresourcetype hipResourcetype_enumToCUresourcetype(
    hipResourcetype resType) {
    switch (resType) {
        case HIP_RESOURCE_TYPE_ARRAY:
            return CU_RESOURCE_TYPE_ARRAY;
        case HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:
            return CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        case HIP_RESOURCE_TYPE_LINEAR:
            return CU_RESOURCE_TYPE_LINEAR;
        case HIP_RESOURCE_TYPE_PITCH2D:
            return CU_RESOURCE_TYPE_PITCH2D;
        default:
            return CU_RESOURCE_TYPE_ARRAY;
    }
}

// hipStreamPerThread
#define hipStreamPerThread ((cudaStream_t)2)

#define hipTexRef CUtexref
#define hiparray CUarray
typedef CUmipmappedArray hipmipmappedArray;
typedef cudaMipmappedArray_t hipMipmappedArray_t;

#define HIP_TRSA_OVERRIDE_FORMAT        CU_TRSA_OVERRIDE_FORMAT
#define HIP_TRSF_READ_AS_INTEGER        CU_TRSF_READ_AS_INTEGER
#define HIP_TRSF_NORMALIZED_COORDINATES CU_TRSF_NORMALIZED_COORDINATES
#define HIP_TRSF_SRGB                   CU_TRSF_SRGB

// hipTextureAddressMode
typedef enum cudaTextureAddressMode hipTextureAddressMode;
#define hipAddressModeWrap cudaAddressModeWrap
#define hipAddressModeClamp cudaAddressModeClamp
#define hipAddressModeMirror cudaAddressModeMirror
#define hipAddressModeBorder cudaAddressModeBorder

// hipTextureFilterMode
typedef enum cudaTextureFilterMode hipTextureFilterMode;
#define hipFilterModePoint cudaFilterModePoint
#define hipFilterModeLinear cudaFilterModeLinear

// hipTextureReadMode
typedef enum cudaTextureReadMode hipTextureReadMode;
#define hipReadModeElementType cudaReadModeElementType
#define hipReadModeNormalizedFloat cudaReadModeNormalizedFloat

// hipChannelFormatKind
typedef enum cudaChannelFormatKind hipChannelFormatKind;
#define hipChannelFormatKindSigned      cudaChannelFormatKindSigned
#define hipChannelFormatKindUnsigned    cudaChannelFormatKindUnsigned
#define hipChannelFormatKindFloat       cudaChannelFormatKindFloat
#define hipChannelFormatKindNone        cudaChannelFormatKindNone

// hipMemRangeAttribute
typedef enum cudaMemRangeAttribute hipMemRangeAttribute;
#define hipMemRangeAttributeReadMostly cudaMemRangeAttributeReadMostly
#define hipMemRangeAttributePreferredLocation cudaMemRangeAttributePreferredLocation
#define hipMemRangeAttributeAccessedBy cudaMemRangeAttributeAccessedBy
#define hipMemRangeAttributeLastPrefetchLocation cudaMemRangeAttributeLastPrefetchLocation

#define hipSurfaceBoundaryMode cudaSurfaceBoundaryMode
#define hipBoundaryModeZero cudaBoundaryModeZero
#define hipBoundaryModeTrap cudaBoundaryModeTrap
#define hipBoundaryModeClamp cudaBoundaryModeClamp

// hipFuncCache
#define hipFuncCachePreferNone cudaFuncCachePreferNone
#define hipFuncCachePreferShared cudaFuncCachePreferShared
#define hipFuncCachePreferL1 cudaFuncCachePreferL1
#define hipFuncCachePreferEqual cudaFuncCachePreferEqual

// hipResourceType
#define hipResourceType cudaResourceType
#define hipResourceTypeArray cudaResourceTypeArray
#define hipResourceTypeMipmappedArray cudaResourceTypeMipmappedArray
#define hipResourceTypeLinear cudaResourceTypeLinear
#define hipResourceTypePitch2D cudaResourceTypePitch2D
//
// hipErrorNoDevice.

// hipResourceViewFormat
typedef enum cudaResourceViewFormat hipResourceViewFormat;
#define hipResViewFormatNone cudaResViewFormatNone
#define hipResViewFormatUnsignedChar1 cudaResViewFormatUnsignedChar1
#define hipResViewFormatUnsignedChar2 cudaResViewFormatUnsignedChar2
#define hipResViewFormatUnsignedChar4 cudaResViewFormatUnsignedChar4
#define hipResViewFormatSignedChar1 cudaResViewFormatSignedChar1
#define hipResViewFormatSignedChar2 cudaResViewFormatSignedChar2
#define hipResViewFormatSignedChar4 cudaResViewFormatSignedChar4
#define hipResViewFormatUnsignedShort1 cudaResViewFormatUnsignedShort1
#define hipResViewFormatUnsignedShort2 cudaResViewFormatUnsignedShort2
#define hipResViewFormatUnsignedShort4 cudaResViewFormatUnsignedShort4
#define hipResViewFormatSignedShort1 cudaResViewFormatSignedShort1
#define hipResViewFormatSignedShort2 cudaResViewFormatSignedShort2
#define hipResViewFormatSignedShort4 cudaResViewFormatSignedShort4
#define hipResViewFormatUnsignedInt1 cudaResViewFormatUnsignedInt1
#define hipResViewFormatUnsignedInt2 cudaResViewFormatUnsignedInt2
#define hipResViewFormatUnsignedInt4 cudaResViewFormatUnsignedInt4
#define hipResViewFormatSignedInt1 cudaResViewFormatSignedInt1
#define hipResViewFormatSignedInt2 cudaResViewFormatSignedInt2
#define hipResViewFormatSignedInt4 cudaResViewFormatSignedInt4
#define hipResViewFormatHalf1 cudaResViewFormatHalf1
#define hipResViewFormatHalf2 cudaResViewFormatHalf2
#define hipResViewFormatHalf4 cudaResViewFormatHalf4
#define hipResViewFormatFloat1 cudaResViewFormatFloat1
#define hipResViewFormatFloat2 cudaResViewFormatFloat2
#define hipResViewFormatFloat4 cudaResViewFormatFloat4
#define hipResViewFormatUnsignedBlockCompressed1 cudaResViewFormatUnsignedBlockCompressed1
#define hipResViewFormatUnsignedBlockCompressed2 cudaResViewFormatUnsignedBlockCompressed2
#define hipResViewFormatUnsignedBlockCompressed3 cudaResViewFormatUnsignedBlockCompressed3
#define hipResViewFormatUnsignedBlockCompressed4 cudaResViewFormatUnsignedBlockCompressed4
#define hipResViewFormatSignedBlockCompressed4 cudaResViewFormatSignedBlockCompressed4
#define hipResViewFormatUnsignedBlockCompressed5 cudaResViewFormatUnsignedBlockCompressed5
#define hipResViewFormatSignedBlockCompressed5 cudaResViewFormatSignedBlockCompressed5
#define hipResViewFormatUnsignedBlockCompressed6H cudaResViewFormatUnsignedBlockCompressed6H
#define hipResViewFormatSignedBlockCompressed6H cudaResViewFormatSignedBlockCompressed6H
#define hipResViewFormatUnsignedBlockCompressed7 cudaResViewFormatUnsignedBlockCompressed7

//! Flags that can be used with hipEventCreateWithFlags:
#define hipEventDefault cudaEventDefault
#define hipEventBlockingSync cudaEventBlockingSync
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventInterprocess cudaEventInterprocess
#define hipEventReleaseToDevice 0 /* no-op on CUDA platform */
#define hipEventReleaseToSystem 0 /* no-op on CUDA platform */


#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostMallocPortable cudaHostAllocPortable
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostMallocWriteCombined cudaHostAllocWriteCombined
#define hipHostMallocCoherent 0x0
#define hipHostMallocNonCoherent 0x0

#define hipMemAttachGlobal cudaMemAttachGlobal
#define hipMemAttachHost cudaMemAttachHost
#define hipMemAttachSingle cudaMemAttachSingle

#define hipHostRegisterDefault cudaHostRegisterDefault
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostRegisterIoMemory cudaHostRegisterIoMemory
#define hipHostRegisterReadOnly cudaHostRegisterReadOnly

#define HIP_LAUNCH_PARAM_BUFFER_POINTER CU_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_SIZE CU_LAUNCH_PARAM_BUFFER_SIZE
#define HIP_LAUNCH_PARAM_END CU_LAUNCH_PARAM_END
#define hipLimitPrintfFifoSize cudaLimitPrintfFifoSize
#define hipLimitMallocHeapSize cudaLimitMallocHeapSize
#define hipLimitStackSize      cudaLimitStackSize
#define hipIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

#define hipOccupancyDefault cudaOccupancyDefault
#define hipOccupancyDisableCachingOverride cudaOccupancyDisableCachingOverride

#define hipCooperativeLaunchMultiDeviceNoPreSync    \
        cudaCooperativeLaunchMultiDeviceNoPreSync
#define hipCooperativeLaunchMultiDeviceNoPostSync   \
        cudaCooperativeLaunchMultiDeviceNoPostSync


// enum CUjit_option redefines
#define HIPRTC_JIT_MAX_REGISTERS CU_JIT_MAX_REGISTERS
#define HIPRTC_JIT_THREADS_PER_BLOCK CU_JIT_THREADS_PER_BLOCK
#define HIPRTC_JIT_WALL_TIME CU_JIT_WALL_TIME
#define HIPRTC_JIT_INFO_LOG_BUFFER CU_JIT_INFO_LOG_BUFFER
#define HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define HIPRTC_JIT_ERROR_LOG_BUFFER CU_JIT_ERROR_LOG_BUFFER
#define HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define HIPRTC_JIT_OPTIMIZATION_LEVEL CU_JIT_OPTIMIZATION_LEVEL
#define HIPRTC_JIT_TARGET_FROM_HIPCONTEXT CU_JIT_TARGET_FROM_CUCONTEXT
#define HIPRTC_JIT_TARGET CU_JIT_TARGET
#define HIPRTC_JIT_FALLBACK_STRATEGY CU_JIT_FALLBACK_STRATEGY
#define HIPRTC_JIT_GENERATE_DEBUG_INFO CU_JIT_GENERATE_DEBUG_INFO
#define HIPRTC_JIT_LOG_VERBOSE CU_JIT_LOG_VERBOSE
#define HIPRTC_JIT_GENERATE_LINE_INFO CU_JIT_GENERATE_LINE_INFO
#define HIPRTC_JIT_CACHE_MODE CU_JIT_CACHE_MODE
#define HIPRTC_JIT_NEW_SM3X_OPT CU_JIT_NEW_SM3X_OPT
#define HIPRTC_JIT_FAST_COMPILE CU_JIT_FAST_COMPILE
#define HIPRTC_JIT_NUM_OPTIONS CU_JIT_NUM_OPTIONS

typedef cudaEvent_t hipEvent_t;
typedef cudaStream_t hipStream_t;
typedef cudaIpcEventHandle_t hipIpcEventHandle_t;
typedef cudaIpcMemHandle_t hipIpcMemHandle_t;
typedef enum cudaLimit hipLimit_t;
typedef enum cudaFuncAttribute hipFuncAttribute;
typedef enum cudaFuncCache hipFuncCache_t;
typedef CUcontext hipCtx_t;
typedef enum cudaSharedMemConfig hipSharedMemConfig;
typedef CUfunc_cache hipFuncCache;
typedef CUjit_option hipJitOption;
typedef CUdevice hipDevice_t;
typedef enum cudaDeviceP2PAttr hipDeviceP2PAttr;
#define hipDevP2PAttrPerformanceRank cudaDevP2PAttrPerformanceRank
#define hipDevP2PAttrAccessSupported cudaDevP2PAttrAccessSupported
#define hipDevP2PAttrNativeAtomicSupported cudaDevP2PAttrNativeAtomicSupported
#define hipDevP2PAttrHipArrayAccessSupported cudaDevP2PAttrCudaArrayAccessSupported
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define hipFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributePreferredSharedMemoryCarveout

typedef CUmodule hipModule_t;
typedef CUfunction hipFunction_t;
typedef CUdeviceptr hipDeviceptr_t;
typedef struct cudaArray hipArray;
typedef struct cudaArray* hipArray_t;
typedef struct cudaArray* hipArray_const_t;
typedef struct cudaFuncAttributes hipFuncAttributes;
typedef struct cudaLaunchParams hipLaunchParams;
typedef CUDA_LAUNCH_PARAMS hipFunctionLaunchParams;
#define hipFunction_attribute CUfunction_attribute
#define hipPointer_attribute CUpointer_attribute
#define hip_Memcpy2D CUDA_MEMCPY2D
#define HIP_MEMCPY3D CUDA_MEMCPY3D
#define hipMemcpy3DParms cudaMemcpy3DParms
#define hipArrayDefault cudaArrayDefault
#define hipArrayLayered cudaArrayLayered
#define hipArraySurfaceLoadStore cudaArraySurfaceLoadStore
#define hipArrayCubemap cudaArrayCubemap
#define hipArrayTextureGather cudaArrayTextureGather

typedef cudaTextureObject_t hipTextureObject_t;
typedef cudaSurfaceObject_t hipSurfaceObject_t;
#define hipTextureType1D cudaTextureType1D
#define hipTextureType1DLayered cudaTextureType1DLayered
#define hipTextureType2D cudaTextureType2D
#define hipTextureType2DLayered cudaTextureType2DLayered
#define hipTextureType3D cudaTextureType3D

#define hipDeviceScheduleAuto cudaDeviceScheduleAuto
#define hipDeviceScheduleSpin cudaDeviceScheduleSpin
#define hipDeviceScheduleYield cudaDeviceScheduleYield
#define hipDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define hipDeviceScheduleMask cudaDeviceScheduleMask
#define hipDeviceMapHost cudaDeviceMapHost
#define hipDeviceLmemResizeToMax cudaDeviceLmemResizeToMax

#define hipCpuDeviceId cudaCpuDeviceId
#define hipInvalidDeviceId cudaInvalidDeviceId
typedef struct cudaExtent hipExtent;
typedef struct cudaPitchedPtr hipPitchedPtr;
typedef struct cudaPos hipPos;
#define make_hipExtent make_cudaExtent
#define make_hipPos make_cudaPos
#define make_hipPitchedPtr make_cudaPitchedPtr
// Flags that can be used with hipStreamCreateWithFlags_cu4oro
#define hipStreamDefault cudaStreamDefault
#define hipStreamNonBlocking cudaStreamNonBlocking

typedef cudaMemPool_t hipMemPool_t;
typedef enum cudaMemPoolAttr hipMemPoolAttr;
#define hipMemPoolReuseFollowEventDependencies cudaMemPoolReuseFollowEventDependencies
#define hipMemPoolReuseAllowOpportunistic cudaMemPoolReuseAllowOpportunistic
#define hipMemPoolReuseAllowInternalDependencies cudaMemPoolReuseAllowInternalDependencies
#define hipMemPoolAttrReleaseThreshold cudaMemPoolAttrReleaseThreshold
#define hipMemPoolAttrReservedMemCurrent cudaMemPoolAttrReservedMemCurrent
#define hipMemPoolAttrReservedMemHigh cudaMemPoolAttrReservedMemHigh
#define hipMemPoolAttrUsedMemCurrent cudaMemPoolAttrUsedMemCurrent
#define hipMemPoolAttrUsedMemHigh cudaMemPoolAttrUsedMemHigh
typedef struct cudaMemLocation hipMemLocation;
typedef struct cudaMemPoolProps hipMemPoolProps;
typedef struct cudaMemAccessDesc hipMemAccessDesc;
typedef enum cudaMemAccessFlags hipMemAccessFlags;
#define hipMemAccessFlagsProtNone cudaMemAccessFlagsProtNone
#define hipMemAccessFlagsProtRead cudaMemAccessFlagsProtRead
#define hipMemAccessFlagsProtReadWrite cudaMemAccessFlagsProtReadWrite
typedef enum cudaMemAllocationHandleType hipMemAllocationHandleType;
typedef struct cudaMemPoolPtrExportData hipMemPoolPtrExportData;

typedef struct cudaChannelFormatDesc hipChannelFormatDesc;
typedef struct cudaResourceDesc hipResourceDesc;
typedef struct cudaTextureDesc hipTextureDesc;
typedef struct cudaResourceViewDesc hipResourceViewDesc;
typedef CUDA_RESOURCE_DESC HIP_RESOURCE_DESC;
typedef CUDA_TEXTURE_DESC HIP_TEXTURE_DESC;
typedef CUDA_RESOURCE_VIEW_DESC HIP_RESOURCE_VIEW_DESC;
// adding code for hipmemSharedConfig
#define hipSharedMemBankSizeDefault cudaSharedMemBankSizeDefault
#define hipSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
#define hipSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte

//Function Attributes
#define HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_NUM_REGS CU_FUNC_ATTRIBUTE_NUM_REGS
#define HIP_FUNC_ATTRIBUTE_PTX_VERSION CU_FUNC_ATTRIBUTE_PTX_VERSION
#define HIP_FUNC_ATTRIBUTE_BINARY_VERSION CU_FUNC_ATTRIBUTE_BINARY_VERSION
#define HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define HIP_FUNC_ATTRIBUTE_MAX CU_FUNC_ATTRIBUTE_MAX

//Pointer Attributes
#define HIP_POINTER_ATTRIBUTE_CONTEXT           CU_POINTER_ATTRIBUTE_CONTEXT
#define HIP_POINTER_ATTRIBUTE_MEMORY_TYPE       CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define HIP_POINTER_ATTRIBUTE_DEVICE_POINTER    CU_POINTER_ATTRIBUTE_DEVICE_POINTER
#define HIP_POINTER_ATTRIBUTE_HOST_POINTER      CU_POINTER_ATTRIBUTE_HOST_POINTER
#define HIP_POINTER_ATTRIBUTE_P2P_TOKENS        CU_POINTER_ATTRIBUTE_P2P_TOKENS
#define HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS       CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define HIP_POINTER_ATTRIBUTE_BUFFER_ID         CU_POINTER_ATTRIBUTE_BUFFER_ID
#define HIP_POINTER_ATTRIBUTE_IS_MANAGED        CU_POINTER_ATTRIBUTE_IS_MANAGED
#define HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE  CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE
#define HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define HIP_POINTER_ATTRIBUTE_RANGE_SIZE        CU_POINTER_ATTRIBUTE_RANGE_SIZE
#define HIP_POINTER_ATTRIBUTE_MAPPED            CU_POINTER_ATTRIBUTE_MAPPED
#define HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS      CU_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE

typedef enum cudaGraphInstantiateFlags hipGraphInstantiateFlags;
#define hipGraphInstantiateFlagAutoFreeOnLaunch cudaGraphInstantiateFlagAutoFreeOnLaunch
#define hipGraphInstantiateFlagUpload cudaGraphInstantiateFlagUpload
#define hipGraphInstantiateFlagDeviceLaunch cudaGraphInstantiateFlagDeviceLaunch
#define hipGraphInstantiateFlagUseNodePriority cudaGraphInstantiateFlagUseNodePriority

#if CUDA_VERSION >= CUDA_9000
#define __shfl(...)      __shfl_sync(0xffffffff, __VA_ARGS__)
#define __shfl_up(...)   __shfl_up_sync(0xffffffff, __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(0xffffffff, __VA_ARGS__)
#define __shfl_xor(...)  __shfl_xor_sync(0xffffffff, __VA_ARGS__)
#endif // CUDA_VERSION >= CUDA_9000

inline static hipError_t hipCUDAErrorTohipError(cudaError_t cuError) {
    switch (cuError) {
        case cudaSuccess:
            return hipSuccess;
        case cudaErrorProfilerDisabled:
            return hipErrorProfilerDisabled;
        case cudaErrorProfilerNotInitialized:
            return hipErrorProfilerNotInitialized;
        case cudaErrorProfilerAlreadyStarted:
            return hipErrorProfilerAlreadyStarted;
        case cudaErrorProfilerAlreadyStopped:
            return hipErrorProfilerAlreadyStopped;
        case cudaErrorInsufficientDriver:
            return hipErrorInsufficientDriver;
        case cudaErrorUnsupportedLimit:
            return hipErrorUnsupportedLimit;
        case cudaErrorPeerAccessUnsupported:
            return hipErrorPeerAccessUnsupported;
        case cudaErrorInvalidGraphicsContext:
            return hipErrorInvalidGraphicsContext;
        case cudaErrorSharedObjectSymbolNotFound:
            return hipErrorSharedObjectSymbolNotFound;
        case cudaErrorSharedObjectInitFailed:
            return hipErrorSharedObjectInitFailed;
        case cudaErrorOperatingSystem:
            return hipErrorOperatingSystem;
        case cudaErrorIllegalState:
            return hipErrorIllegalState;
        case cudaErrorSetOnActiveProcess:
            return hipErrorSetOnActiveProcess;
        case cudaErrorIllegalAddress:
            return hipErrorIllegalAddress;
        case cudaErrorInvalidSymbol:
            return hipErrorInvalidSymbol;
        case cudaErrorMissingConfiguration:
            return hipErrorMissingConfiguration;
        case cudaErrorMemoryAllocation:
            return hipErrorOutOfMemory;
        case cudaErrorInitializationError:
            return hipErrorNotInitialized;
        case cudaErrorLaunchFailure:
            return hipErrorLaunchFailure;
        case cudaErrorCooperativeLaunchTooLarge:
            return hipErrorCooperativeLaunchTooLarge;
        case cudaErrorPriorLaunchFailure:
            return hipErrorPriorLaunchFailure;
        case cudaErrorLaunchOutOfResources:
            return hipErrorLaunchOutOfResources;
        case cudaErrorInvalidDeviceFunction:
            return hipErrorInvalidDeviceFunction;
        case cudaErrorInvalidConfiguration:
            return hipErrorInvalidConfiguration;
        case cudaErrorInvalidDevice:
            return hipErrorInvalidDevice;
        case cudaErrorInvalidValue:
            return hipErrorInvalidValue;
        case cudaErrorInvalidPitchValue:
            return hipErrorInvalidPitchValue;
        case cudaErrorInvalidDevicePointer:
            return hipErrorInvalidDevicePointer;
        case cudaErrorInvalidMemcpyDirection:
            return hipErrorInvalidMemcpyDirection;
        case cudaErrorInvalidResourceHandle:
            return hipErrorInvalidHandle;
        case cudaErrorNotReady:
            return hipErrorNotReady;
        case cudaErrorNoDevice:
            return hipErrorNoDevice;
        case cudaErrorPeerAccessAlreadyEnabled:
            return hipErrorPeerAccessAlreadyEnabled;
        case cudaErrorPeerAccessNotEnabled:
            return hipErrorPeerAccessNotEnabled;
        case cudaErrorContextIsDestroyed:
            return hipErrorContextIsDestroyed;
        case cudaErrorHostMemoryAlreadyRegistered:
            return hipErrorHostMemoryAlreadyRegistered;
        case cudaErrorHostMemoryNotRegistered:
            return hipErrorHostMemoryNotRegistered;
        case cudaErrorMapBufferObjectFailed:
            return hipErrorMapFailed;
        case cudaErrorAssert:
            return hipErrorAssert;
        case cudaErrorNotSupported:
            return hipErrorNotSupported;
        case cudaErrorCudartUnloading:
            return hipErrorDeinitialized;
        case cudaErrorInvalidKernelImage:
            return hipErrorInvalidImage;
        case cudaErrorUnmapBufferObjectFailed:
            return hipErrorUnmapFailed;
        case cudaErrorNoKernelImageForDevice:
            return hipErrorNoBinaryForGpu;
        case cudaErrorECCUncorrectable:
            return hipErrorECCNotCorrectable;
        case cudaErrorDeviceAlreadyInUse:
            return hipErrorContextAlreadyInUse;
        case cudaErrorInvalidPtx:
            return hipErrorInvalidKernelFile;
        case cudaErrorLaunchTimeout:
            return hipErrorLaunchTimeOut;
#if CUDA_VERSION >= CUDA_10010
        case cudaErrorInvalidSource:
            return hipErrorInvalidSource;
        case cudaErrorFileNotFound:
            return hipErrorFileNotFound;
        case cudaErrorSymbolNotFound:
            return hipErrorNotFound;
        case cudaErrorArrayIsMapped:
            return hipErrorArrayIsMapped;
        case cudaErrorNotMappedAsPointer:
            return hipErrorNotMappedAsPointer;
        case cudaErrorNotMappedAsArray:
            return hipErrorNotMappedAsArray;
        case cudaErrorNotMapped:
            return hipErrorNotMapped;
        case cudaErrorAlreadyAcquired:
            return hipErrorAlreadyAcquired;
        case cudaErrorAlreadyMapped:
            return hipErrorAlreadyMapped;
#endif
#if CUDA_VERSION >= CUDA_10020
        case cudaErrorDeviceUninitialized:
            return hipErrorInvalidContext;
#endif
        case cudaErrorStreamCaptureUnsupported:
            return hipErrorStreamCaptureUnsupported;
        case cudaErrorStreamCaptureInvalidated:
            return hipErrorStreamCaptureInvalidated;
        case cudaErrorStreamCaptureMerge:
            return hipErrorStreamCaptureMerge;
        case cudaErrorStreamCaptureUnmatched:
            return hipErrorStreamCaptureUnmatched;
        case cudaErrorStreamCaptureUnjoined:
            return hipErrorStreamCaptureUnjoined;
        case cudaErrorStreamCaptureIsolation:
            return hipErrorStreamCaptureIsolation;
        case cudaErrorStreamCaptureImplicit:
            return hipErrorStreamCaptureImplicit;
        case cudaErrorCapturedEvent:
            return hipErrorCapturedEvent;
        case cudaErrorStreamCaptureWrongThread:
            return hipErrorStreamCaptureWrongThread;
        case cudaErrorGraphExecUpdateFailure:
            return hipErrorGraphExecUpdateFailure;
        case cudaErrorUnknown:
        default:
            return hipErrorUnknown;  // Note - translated error.
    }
}

inline static hipError_t hipCUResultTohipError(CUresult cuError) {
    switch (cuError) {
        case CUDA_SUCCESS:
            return hipSuccess;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return hipErrorOutOfMemory;
        case CUDA_ERROR_INVALID_VALUE:
            return hipErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE:
            return hipErrorInvalidDevice;
        case CUDA_ERROR_DEINITIALIZED:
            return hipErrorDeinitialized;
        case CUDA_ERROR_NO_DEVICE:
            return hipErrorNoDevice;
        case CUDA_ERROR_INVALID_CONTEXT:
            return hipErrorInvalidContext;
        case CUDA_ERROR_NOT_INITIALIZED:
            return hipErrorNotInitialized;
        case CUDA_ERROR_INVALID_HANDLE:
            return hipErrorInvalidHandle;
        case CUDA_ERROR_MAP_FAILED:
            return hipErrorMapFailed;
        case CUDA_ERROR_PROFILER_DISABLED:
            return hipErrorProfilerDisabled;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return hipErrorProfilerNotInitialized;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return hipErrorProfilerAlreadyStarted;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return hipErrorProfilerAlreadyStopped;
        case CUDA_ERROR_INVALID_IMAGE:
            return hipErrorInvalidImage;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return hipErrorContextAlreadyCurrent;
        case CUDA_ERROR_UNMAP_FAILED:
            return hipErrorUnmapFailed;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return hipErrorArrayIsMapped;
        case CUDA_ERROR_ALREADY_MAPPED:
            return hipErrorAlreadyMapped;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return hipErrorNoBinaryForGpu;
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return hipErrorAlreadyAcquired;
        case CUDA_ERROR_NOT_MAPPED:
            return hipErrorNotMapped;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return hipErrorNotMappedAsArray;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return hipErrorNotMappedAsPointer;
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return hipErrorECCNotCorrectable;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return hipErrorUnsupportedLimit;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return hipErrorContextAlreadyInUse;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return hipErrorPeerAccessUnsupported;
        case CUDA_ERROR_INVALID_PTX:
            return hipErrorInvalidKernelFile;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return hipErrorInvalidGraphicsContext;
        case CUDA_ERROR_INVALID_SOURCE:
            return hipErrorInvalidSource;
        case CUDA_ERROR_FILE_NOT_FOUND:
            return hipErrorFileNotFound;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return hipErrorSharedObjectSymbolNotFound;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return hipErrorSharedObjectInitFailed;
        case CUDA_ERROR_OPERATING_SYSTEM:
            return hipErrorOperatingSystem;
        case CUDA_ERROR_ILLEGAL_STATE:
            return hipErrorIllegalState;
        case CUDA_ERROR_NOT_FOUND:
            return hipErrorNotFound;
        case CUDA_ERROR_NOT_READY:
            return hipErrorNotReady;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return hipErrorIllegalAddress;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return hipErrorLaunchOutOfResources;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return hipErrorLaunchTimeOut;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return hipErrorPeerAccessAlreadyEnabled;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return hipErrorPeerAccessNotEnabled;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return hipErrorSetOnActiveProcess;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return hipErrorContextIsDestroyed;
        case CUDA_ERROR_ASSERT:
            return hipErrorAssert;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return hipErrorHostMemoryAlreadyRegistered;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return hipErrorHostMemoryNotRegistered;
        case CUDA_ERROR_LAUNCH_FAILED:
            return hipErrorLaunchFailure;
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return hipErrorCooperativeLaunchTooLarge;
        case CUDA_ERROR_NOT_SUPPORTED:
            return hipErrorNotSupported;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            return hipErrorStreamCaptureUnsupported;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            return hipErrorStreamCaptureInvalidated;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            return hipErrorStreamCaptureMerge;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            return hipErrorStreamCaptureUnmatched;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            return hipErrorStreamCaptureUnjoined;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            return hipErrorStreamCaptureIsolation;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            return hipErrorStreamCaptureImplicit;
        case CUDA_ERROR_CAPTURED_EVENT:
            return hipErrorCapturedEvent;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            return hipErrorStreamCaptureWrongThread;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            return hipErrorGraphExecUpdateFailure;
        case CUDA_ERROR_UNKNOWN:
        default:
            return hipErrorUnknown;  // Note - translated error.
    }
}

inline static CUresult hipErrorToCUResult(hipError_t hError) {
    switch (hError) {
        case hipSuccess:
            return CUDA_SUCCESS;
        case hipErrorOutOfMemory:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case hipErrorInvalidValue:
            return CUDA_ERROR_INVALID_VALUE;
        case hipErrorInvalidDevice:
            return CUDA_ERROR_INVALID_DEVICE;
        case hipErrorDeinitialized:
            return CUDA_ERROR_DEINITIALIZED;
        case hipErrorNoDevice:
            return CUDA_ERROR_NO_DEVICE;
        case hipErrorInvalidContext:
            return CUDA_ERROR_INVALID_CONTEXT;
        case hipErrorNotInitialized:
            return CUDA_ERROR_NOT_INITIALIZED;
        case hipErrorInvalidHandle:
            return CUDA_ERROR_INVALID_HANDLE;
        case hipErrorMapFailed:
            return CUDA_ERROR_MAP_FAILED;
        case hipErrorProfilerDisabled:
            return CUDA_ERROR_PROFILER_DISABLED;
        case hipErrorProfilerNotInitialized:
            return CUDA_ERROR_PROFILER_NOT_INITIALIZED;
        case hipErrorProfilerAlreadyStarted:
            return CUDA_ERROR_PROFILER_ALREADY_STARTED;
        case hipErrorProfilerAlreadyStopped:
            return CUDA_ERROR_PROFILER_ALREADY_STOPPED;
        case hipErrorInvalidImage:
            return CUDA_ERROR_INVALID_IMAGE;
        case hipErrorContextAlreadyCurrent:
            return CUDA_ERROR_CONTEXT_ALREADY_CURRENT;
        case hipErrorUnmapFailed:
            return CUDA_ERROR_UNMAP_FAILED;
        case hipErrorArrayIsMapped:
            return CUDA_ERROR_ARRAY_IS_MAPPED;
        case hipErrorAlreadyMapped:
            return CUDA_ERROR_ALREADY_MAPPED;
        case hipErrorNoBinaryForGpu:
            return CUDA_ERROR_NO_BINARY_FOR_GPU;
        case hipErrorAlreadyAcquired:
            return CUDA_ERROR_ALREADY_ACQUIRED;
        case hipErrorNotMapped:
            return CUDA_ERROR_NOT_MAPPED;
        case hipErrorNotMappedAsArray:
            return CUDA_ERROR_NOT_MAPPED_AS_ARRAY;
        case hipErrorNotMappedAsPointer:
            return CUDA_ERROR_NOT_MAPPED_AS_POINTER;
        case hipErrorECCNotCorrectable:
            return CUDA_ERROR_ECC_UNCORRECTABLE;
        case hipErrorUnsupportedLimit:
            return CUDA_ERROR_UNSUPPORTED_LIMIT;
        case hipErrorContextAlreadyInUse:
            return CUDA_ERROR_CONTEXT_ALREADY_IN_USE;
        case hipErrorPeerAccessUnsupported:
            return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
        case hipErrorInvalidKernelFile:
            return CUDA_ERROR_INVALID_PTX;
        case hipErrorInvalidGraphicsContext:
            return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
        case hipErrorInvalidSource:
            return CUDA_ERROR_INVALID_SOURCE;
        case hipErrorFileNotFound:
            return CUDA_ERROR_FILE_NOT_FOUND;
        case hipErrorSharedObjectSymbolNotFound:
            return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
        case hipErrorSharedObjectInitFailed:
            return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
        case hipErrorOperatingSystem:
            return CUDA_ERROR_OPERATING_SYSTEM;
        case hipErrorIllegalState:
            return CUDA_ERROR_ILLEGAL_STATE;
        case hipErrorNotFound:
            return CUDA_ERROR_NOT_FOUND;
        case hipErrorNotReady:
            return CUDA_ERROR_NOT_READY;
        case hipErrorIllegalAddress:
            return CUDA_ERROR_ILLEGAL_ADDRESS;
        case hipErrorLaunchOutOfResources:
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        case hipErrorLaunchTimeOut:
            return CUDA_ERROR_LAUNCH_TIMEOUT;
        case hipErrorPeerAccessAlreadyEnabled:
            return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
        case hipErrorPeerAccessNotEnabled:
            return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
        case hipErrorSetOnActiveProcess:
            return CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
        case hipErrorContextIsDestroyed:
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        case hipErrorAssert:
            return CUDA_ERROR_ASSERT;
        case hipErrorHostMemoryAlreadyRegistered:
            return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
        case hipErrorHostMemoryNotRegistered:
            return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
        case hipErrorLaunchFailure:
            return CUDA_ERROR_LAUNCH_FAILED;
        case hipErrorCooperativeLaunchTooLarge:
            return CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
        case hipErrorNotSupported:
            return CUDA_ERROR_NOT_SUPPORTED;
        case hipErrorStreamCaptureUnsupported:
            return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        case hipErrorStreamCaptureInvalidated:
            return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
        case hipErrorStreamCaptureMerge:
            return CUDA_ERROR_STREAM_CAPTURE_MERGE;
        case hipErrorStreamCaptureUnmatched:
            return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
        case hipErrorStreamCaptureUnjoined:
            return CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
        case hipErrorStreamCaptureIsolation:
            return CUDA_ERROR_STREAM_CAPTURE_ISOLATION;
        case hipErrorStreamCaptureImplicit:
            return CUDA_ERROR_STREAM_CAPTURE_IMPLICIT;
        case hipErrorCapturedEvent:
            return CUDA_ERROR_CAPTURED_EVENT;
        case hipErrorStreamCaptureWrongThread:
            return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
        case hipErrorGraphExecUpdateFailure:
            return CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
        case hipErrorUnknown:
        default:
            return CUDA_ERROR_UNKNOWN;  // Note - translated error.
    }
}

inline static cudaError_t hipErrorToCudaError(hipError_t hError) {
    switch (hError) {
        case hipSuccess:
            return cudaSuccess;
        case hipErrorOutOfMemory:
            return cudaErrorMemoryAllocation;
        case hipErrorProfilerDisabled:
          return cudaErrorProfilerDisabled;
        case hipErrorProfilerNotInitialized:
            return cudaErrorProfilerNotInitialized;
        case hipErrorProfilerAlreadyStarted:
            return cudaErrorProfilerAlreadyStarted;
        case hipErrorProfilerAlreadyStopped:
            return cudaErrorProfilerAlreadyStopped;
        case hipErrorInvalidConfiguration:
            return cudaErrorInvalidConfiguration;
        case hipErrorLaunchOutOfResources:
            return cudaErrorLaunchOutOfResources;
        case hipErrorInvalidValue:
            return cudaErrorInvalidValue;
        case hipErrorInvalidPitchValue:
            return cudaErrorInvalidPitchValue;
        case hipErrorInvalidHandle:
            return cudaErrorInvalidResourceHandle;
        case hipErrorInvalidDevice:
            return cudaErrorInvalidDevice;
        case hipErrorInvalidMemcpyDirection:
            return cudaErrorInvalidMemcpyDirection;
        case hipErrorInvalidDevicePointer:
            return cudaErrorInvalidDevicePointer;
        case hipErrorNotInitialized:
            return cudaErrorInitializationError;
        case hipErrorNoDevice:
            return cudaErrorNoDevice;
        case hipErrorNotReady:
            return cudaErrorNotReady;
        case hipErrorPeerAccessNotEnabled:
            return cudaErrorPeerAccessNotEnabled;
        case hipErrorPeerAccessAlreadyEnabled:
            return cudaErrorPeerAccessAlreadyEnabled;
        case hipErrorHostMemoryAlreadyRegistered:
            return cudaErrorHostMemoryAlreadyRegistered;
        case hipErrorHostMemoryNotRegistered:
            return cudaErrorHostMemoryNotRegistered;
        case hipErrorDeinitialized:
            return cudaErrorCudartUnloading;
        case hipErrorInvalidSymbol:
            return cudaErrorInvalidSymbol;
        case hipErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
        case hipErrorMissingConfiguration:
            return cudaErrorMissingConfiguration;
        case hipErrorPriorLaunchFailure:
            return cudaErrorPriorLaunchFailure;
        case hipErrorInvalidDeviceFunction:
            return cudaErrorInvalidDeviceFunction;
        case hipErrorInvalidImage:
            return cudaErrorInvalidKernelImage;
        case hipErrorInvalidContext:
#if CUDA_VERSION >= CUDA_10020
            return cudaErrorDeviceUninitialized;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorMapFailed:
            return cudaErrorMapBufferObjectFailed;
        case hipErrorUnmapFailed:
            return cudaErrorUnmapBufferObjectFailed;
        case hipErrorArrayIsMapped:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorArrayIsMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorAlreadyMapped:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorAlreadyMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNoBinaryForGpu:
            return cudaErrorNoKernelImageForDevice;
        case hipErrorAlreadyAcquired:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorAlreadyAcquired;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMapped:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorNotMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMappedAsArray:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorNotMappedAsArray;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMappedAsPointer:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorNotMappedAsPointer;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorECCNotCorrectable:
            return cudaErrorECCUncorrectable;
        case hipErrorUnsupportedLimit:
            return cudaErrorUnsupportedLimit;
        case hipErrorContextAlreadyInUse:
            return cudaErrorDeviceAlreadyInUse;
        case hipErrorPeerAccessUnsupported:
            return cudaErrorPeerAccessUnsupported;
        case hipErrorInvalidKernelFile:
            return cudaErrorInvalidPtx;
        case hipErrorInvalidGraphicsContext:
            return cudaErrorInvalidGraphicsContext;
        case hipErrorInvalidSource:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorInvalidSource;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorFileNotFound:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorFileNotFound;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorSharedObjectSymbolNotFound:
            return cudaErrorSharedObjectSymbolNotFound;
        case hipErrorSharedObjectInitFailed:
            return cudaErrorSharedObjectInitFailed;
        case hipErrorOperatingSystem:
            return cudaErrorOperatingSystem;
        case hipErrorIllegalState:
            return cudaErrorIllegalState;
        case hipErrorNotFound:
#if CUDA_VERSION >= CUDA_10010
            return cudaErrorSymbolNotFound;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorIllegalAddress:
            return cudaErrorIllegalAddress;
        case hipErrorLaunchTimeOut:
            return cudaErrorLaunchTimeout;
        case hipErrorSetOnActiveProcess:
            return cudaErrorSetOnActiveProcess;
        case hipErrorContextIsDestroyed:
            return cudaErrorContextIsDestroyed;
        case hipErrorAssert:
            return cudaErrorAssert;
        case hipErrorLaunchFailure:
            return cudaErrorLaunchFailure;
        case hipErrorCooperativeLaunchTooLarge:
            return cudaErrorCooperativeLaunchTooLarge;
        case hipErrorStreamCaptureUnsupported:
            return cudaErrorStreamCaptureUnsupported;
        case hipErrorStreamCaptureInvalidated:
            return cudaErrorStreamCaptureInvalidated;
        case hipErrorStreamCaptureMerge:
            return cudaErrorStreamCaptureMerge;
        case hipErrorStreamCaptureUnmatched:
            return cudaErrorStreamCaptureUnmatched;
        case hipErrorStreamCaptureUnjoined:
            return cudaErrorStreamCaptureUnjoined;
        case hipErrorStreamCaptureIsolation:
            return cudaErrorStreamCaptureIsolation;
        case hipErrorStreamCaptureImplicit:
            return cudaErrorStreamCaptureImplicit;
        case hipErrorCapturedEvent:
            return cudaErrorCapturedEvent;
        case hipErrorStreamCaptureWrongThread:
            return cudaErrorStreamCaptureWrongThread;
        case hipErrorGraphExecUpdateFailure:
            return cudaErrorGraphExecUpdateFailure;
        case hipErrorNotSupported:
            return cudaErrorNotSupported;
        // HSA: does not exist in CUDA
        case hipErrorRuntimeMemory:
        // HSA: does not exist in CUDA
        case hipErrorRuntimeOther:
        case hipErrorUnknown:
        case hipErrorTbd:
        default:
            return cudaErrorUnknown;  // Note - translated error.
    }
}

inline static enum cudaMemcpyKind hipMemcpyKindToCudaMemcpyKind(hipMemcpyKind kind) {
    switch (kind) {
        case hipMemcpyHostToHost:
            return cudaMemcpyHostToHost;
        case hipMemcpyHostToDevice:
            return cudaMemcpyHostToDevice;
        case hipMemcpyDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case hipMemcpyDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        case hipMemcpyDefault:
            return cudaMemcpyDefault;
        default:
            return (hipMemcpyKind)-1;
    }
}

inline static enum cudaTextureAddressMode hipTextureAddressModeToCudaTextureAddressMode(
    hipTextureAddressMode kind) {
    switch (kind) {
        case hipAddressModeWrap:
            return cudaAddressModeWrap;
        case hipAddressModeClamp:
            return cudaAddressModeClamp;
        case hipAddressModeMirror:
            return cudaAddressModeMirror;
        case hipAddressModeBorder:
            return cudaAddressModeBorder;
        default:
            return (hipTextureAddressMode)-1;
    }
}

inline static enum cudaMemRangeAttribute hipMemRangeAttributeToCudaMemRangeAttribute(
   hipMemRangeAttribute kind) {
   switch (kind) {
       case hipMemRangeAttributeReadMostly:
           return cudaMemRangeAttributeReadMostly;
       case hipMemRangeAttributePreferredLocation:
           return cudaMemRangeAttributePreferredLocation;
       case hipMemRangeAttributeAccessedBy:
           return cudaMemRangeAttributeAccessedBy;
       case hipMemRangeAttributeLastPrefetchLocation:
           return cudaMemRangeAttributeLastPrefetchLocation;
       default:
           return (hipMemRangeAttribute)-1;
   }
}

inline static enum cudaMemoryAdvise hipMemoryAdviseTocudaMemoryAdvise(
    hipMemoryAdvise kind) {
   switch (kind) {
       case hipMemAdviseSetReadMostly:
           return cudaMemAdviseSetReadMostly;
       case hipMemAdviseUnsetReadMostly :
           return cudaMemAdviseUnsetReadMostly ;
       case hipMemAdviseSetPreferredLocation:
           return cudaMemAdviseSetPreferredLocation;
       case hipMemAdviseUnsetPreferredLocation:
           return cudaMemAdviseUnsetPreferredLocation;
       case hipMemAdviseSetAccessedBy:
           return cudaMemAdviseSetAccessedBy;
       case hipMemAdviseUnsetAccessedBy:
           return cudaMemAdviseUnsetAccessedBy;
       default:
           return (enum cudaMemoryAdvise)-1;
   }
}

inline static enum cudaTextureFilterMode hipTextureFilterModeToCudaTextureFilterMode(
    hipTextureFilterMode kind) {
    switch (kind) {
        case hipFilterModePoint:
            return cudaFilterModePoint;
        case hipFilterModeLinear:
            return cudaFilterModeLinear;
        default:
            return (hipTextureFilterMode)-1;
    }
}

inline static enum cudaTextureReadMode hipTextureReadModeToCudaTextureReadMode(hipTextureReadMode kind) {
    switch (kind) {
        case hipReadModeElementType:
            return cudaReadModeElementType;
        case hipReadModeNormalizedFloat:
            return cudaReadModeNormalizedFloat;
        default:
            return (hipTextureReadMode)-1;
    }
}

inline static enum cudaChannelFormatKind hipChannelFormatKindToCudaChannelFormatKind(
    hipChannelFormatKind kind) {
    switch (kind) {
        case hipChannelFormatKindSigned:
            return cudaChannelFormatKindSigned;
        case hipChannelFormatKindUnsigned:
            return cudaChannelFormatKindUnsigned;
        case hipChannelFormatKindFloat:
            return cudaChannelFormatKindFloat;
        case hipChannelFormatKindNone:
            return cudaChannelFormatKindNone;
        default:
            return (hipChannelFormatKind)-1;
    }
}

typedef enum cudaExternalMemoryHandleType hipExternalMemoryHandleType;
#define hipExternalMemoryHandleTypeOpaqueFd cudaExternalMemoryHandleTypeOpaqueFd
#define hipExternalMemoryHandleTypeOpaqueWin32 cudaExternalMemoryHandleTypeOpaqueWin32
#define hipExternalMemoryHandleTypeOpaqueWin32Kmt cudaExternalMemoryHandleTypeOpaqueWin32Kmt
#define hipExternalMemoryHandleTypeD3D12Heap cudaExternalMemoryHandleTypeD3D12Heap
#define hipExternalMemoryHandleTypeD3D12Resource cudaExternalMemoryHandleTypeD3D12Resource
#if CUDA_VERSION >= CUDA_10020
#define hipExternalMemoryHandleTypeD3D11Resource cudaExternalMemoryHandleTypeD3D11Resource
#define hipExternalMemoryHandleTypeD3D11ResourceKmt cudaExternalMemoryHandleTypeD3D11ResourceKmt
#define hipExternalMemoryHandleTypeNvSciBuf cudaExternalMemoryHandleTypeNvSciBuf
#endif

typedef struct cudaExternalMemoryHandleDesc hipExternalMemoryHandleDesc;
typedef struct cudaExternalMemoryBufferDesc hipExternalMemoryBufferDesc;
typedef cudaExternalMemory_t hipExternalMemory_t;

typedef enum cudaExternalSemaphoreHandleType hipExternalSemaphoreHandleType;
#define hipExternalSemaphoreHandleTypeOpaqueFd cudaExternalSemaphoreHandleTypeOpaqueFd
#define hipExternalSemaphoreHandleTypeOpaqueWin32 cudaExternalSemaphoreHandleTypeOpaqueWin32
#define hipExternalSemaphoreHandleTypeOpaqueWin32Kmt cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define hipExternalSemaphoreHandleTypeD3D12Fence cudaExternalSemaphoreHandleTypeD3D12Fence
#if CUDA_VERSION >= CUDA_10020
#define hipExternalSemaphoreHandleTypeD3D11Fence cudaExternalSemaphoreHandleTypeD3D11Fence
#define hipExternalSemaphoreHandleTypeNvSciSync cudaExternalSemaphoreHandleTypeNvSciSync
#define hipExternalSemaphoreHandleTypeKeyedMutex cudaExternalSemaphoreHandleTypeKeyedMutex
#define hipExternalSemaphoreHandleTypeKeyedMutexKmt cudaExternalSemaphoreHandleTypeKeyedMutexKmt
#endif
#if CUDA_VERSION >= CUDA_11020
#define hipExternalSemaphoreHandleTypeTimelineSemaphoreFd cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
#define hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
#endif

typedef struct cudaExternalSemaphoreHandleDesc hipExternalSemaphoreHandleDesc;
typedef cudaExternalSemaphore_t hipExternalSemaphore_t;
typedef struct cudaExternalSemaphoreSignalParams hipExternalSemaphoreSignalParams;
typedef struct cudaExternalSemaphoreWaitParams hipExternalSemaphoreWaitParams;

typedef struct cudaGraphicsResource hipGraphicsResource;
typedef cudaGraphicsResource_t hipGraphicsResource_t;

typedef enum cudaGraphicsRegisterFlags hipGraphicsRegisterFlags;
#define hipGraphicsRegisterFlagsNone cudaGraphicsRegisterFlagsNone
#define hipGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsReadOnly
#define hipGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsWriteDiscard
#define hipGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsSurfaceLoadStore
#define hipGraphicsRegisterFlagsTextureGather cudaGraphicsRegisterFlagsTextureGather

/**
 * graph types
 *
 */
typedef cudaGraph_t hipGraph_t;
typedef cudaGraphNode_t hipGraphNode_t;
typedef cudaGraphExec_t hipGraphExec_t;
typedef cudaUserObject_t hipUserObject_t;

typedef enum cudaGraphNodeType hipGraphNodeType;
#define hipGraphNodeTypeKernel cudaGraphNodeTypeKernel
#define hipGraphNodeTypeMemcpy cudaGraphNodeTypeMemcpy
#define hipGraphNodeTypeMemset cudaGraphNodeTypeMemset
#define hipGraphNodeTypeHost cudaGraphNodeTypeHost
#define hipGraphNodeTypeGraph cudaGraphNodeTypeGraph
#define hipGraphNodeTypeEmpty cudaGraphNodeTypeEmpty
#define hipGraphNodeTypeWaitEvent cudaGraphNodeTypeWaitEvent
#define hipGraphNodeTypeEventRecord cudaGraphNodeTypeEventRecord
#define hipGraphNodeTypeExtSemaphoreSignal cudaGraphNodeTypeExtSemaphoreSignal
#define hipGraphNodeTypeExtSemaphoreWait  cudaGraphNodeTypeExtSemaphoreWait
#define hipGraphNodeTypeMemcpyFromSymbol cudaGraphNodeTypeMemcpyFromSymbol
#define hipGraphNodeTypeMemcpyToSymbol cudaGraphNodeTypeMemcpyToSymbol
#define hipGraphNodeTypeCount cudaGraphNodeTypeCount

typedef cudaHostFn_t hipHostFn_t;
typedef struct cudaHostNodeParams hipHostNodeParams;
typedef struct cudaKernelNodeParams hipKernelNodeParams;
typedef struct cudaMemsetParams hipMemsetParams;

#if CUDA_VERSION >= CUDA_11040
typedef struct cudaMemAllocNodeParams hipMemAllocNodeParams;
#endif

typedef enum cudaGraphExecUpdateResult hipGraphExecUpdateResult;
#define hipGraphExecUpdateSuccess cudaGraphExecUpdateSuccess
#define hipGraphExecUpdateError cudaGraphExecUpdateError
#define hipGraphExecUpdateErrorTopologyChanged cudaGraphExecUpdateErrorTopologyChanged
#define hipGraphExecUpdateErrorNodeTypeChanged cudaGraphExecUpdateErrorNodeTypeChanged
#define hipGraphExecUpdateErrorFunctionChanged cudaGraphExecUpdateErrorFunctionChanged
#define hipGraphExecUpdateErrorParametersChanged cudaGraphExecUpdateErrorParametersChanged
#define hipGraphExecUpdateErrorNotSupported cudaGraphExecUpdateErrorNotSupported
#define hipGraphExecUpdateErrorUnsupportedFunctionChange                                           \
  cudaGraphExecUpdateErrorUnsupportedFunctionChange

typedef enum cudaStreamCaptureMode hipStreamCaptureMode;
#define hipStreamCaptureModeGlobal cudaStreamCaptureModeGlobal
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
#define hipStreamCaptureModeRelaxed cudaStreamCaptureModeRelaxed

typedef enum cudaStreamCaptureStatus hipStreamCaptureStatus;
#define hipStreamCaptureStatusNone cudaStreamCaptureStatusNone
#define hipStreamCaptureStatusActive cudaStreamCaptureStatusActive
#define hipStreamCaptureStatusInvalidated cudaStreamCaptureStatusInvalidated

typedef union cudaKernelNodeAttrValue hipKernelNodeAttrValue;
typedef enum  cudaKernelNodeAttrID hipKernelNodeAttrID;
#define hipKernelNodeAttributeAccessPolicyWindow cudaKernelNodeAttributeAccessPolicyWindow
#define hipKernelNodeAttributeCooperative cudaKernelNodeAttributeCooperative
typedef enum cudaAccessProperty hipAccessProperty;
#define hipAccessPropertyNormal cudaAccessPropertyNormal
#define hipAccessPropertyStreaming cudaAccessPropertyStreaming
#define hipAccessPropertyPersisting cudaAccessPropertyPersisting
typedef struct cudaAccessPolicyWindow hipAccessPolicyWindow;

typedef enum  cudaGraphMemAttributeType hipGraphMemAttributeType;
#define hipGraphMemAttrUsedMemCurrent cudaGraphMemAttrUsedMemCurrent
#define hipGraphMemAttrUsedMemHigh cudaGraphMemAttrUsedMemHigh
#define hipGraphMemAttrReservedMemCurrent cudaGraphMemAttrReservedMemCurrent
#define hipGraphMemAttrReservedMemHigh cudaGraphMemAttrReservedMemHigh

typedef enum cudaUserObjectFlags hipUserObjectFlags;
#define hipUserObjectNoDestructorSync cudaUserObjectNoDestructorSync

typedef enum cudaUserObjectRetainFlags hipUserObjectRetainFlags;
#define hipGraphUserObjectMove cudaGraphUserObjectMove

#if CUDA_VERSION >= CUDA_11030
typedef enum cudaStreamUpdateCaptureDependenciesFlags hipStreamUpdateCaptureDependenciesFlags;
#define hipStreamAddCaptureDependencies cudaStreamAddCaptureDependencies
#define hipStreamSetCaptureDependencies cudaStreamSetCaptureDependencies
#endif

#if CUDA_VERSION >= CUDA_11030
typedef enum cudaGraphDebugDotFlags hipGraphDebugDotFlags;
#define hipGraphDebugDotFlagsVerbose cudaGraphDebugDotFlagsVerbose
#define hipGraphDebugDotFlagsKernelNodeParams cudaGraphDebugDotFlagsKernelNodeParams
#define hipGraphDebugDotFlagsMemcpyNodeParams cudaGraphDebugDotFlagsMemcpyNodeParams
#define hipGraphDebugDotFlagsMemsetNodeParams cudaGraphDebugDotFlagsMemsetNodeParams
#define hipGraphDebugDotFlagsHostNodeParams cudaGraphDebugDotFlagsHostNodeParams
#define hipGraphDebugDotFlagsEventNodeParams cudaGraphDebugDotFlagsEventNodeParams
#define hipGraphDebugDotFlagsExtSemasSignalNodeParams cudaGraphDebugDotFlagsExtSemasSignalNodeParams
#define hipGraphDebugDotFlagsExtSemasWaitNodeParams cudaGraphDebugDotFlagsExtSemasWaitNodeParams
#define hipGraphDebugDotFlagsKernelNodeAttributes cudaGraphDebugDotFlagsKernelNodeAttributes
#define hipGraphDebugDotFlagsHandles cudaGraphDebugDotFlagsHandles
#endif

#if CUDA_VERSION >= CUDA_10020
#define hipMemAllocationGranularityMinimum CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define hipMemAllocationGranularityRecommended CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
typedef enum CUmemAllocationGranularity_flags_enum  hipMemAllocationGranularity_flags;
typedef enum cudaMemLocationType hipMemLocationType;
#define hipMemLocationTypeInvalid cudaMemLocationTypeInvalid
#define hipMemLocationTypeDevice cudaMemLocationTypeDevice
#define hipMemHandleTypeNone cudaMemHandleTypeNone
#define hipMemHandleTypePosixFileDescriptor cudaMemHandleTypePosixFileDescriptor
#define hipMemHandleTypeWin32 cudaMemHandleTypeWin32
#define hipMemHandleTypeWin32Kmt cudaMemHandleTypeWin32Kmt
typedef enum cudaMemAllocationType hipMemAllocationType;
#define hipMemAllocationTypeInvalid cudaMemAllocationTypeInvalid
#define hipMemAllocationTypePinned cudaMemAllocationTypePinned
#define hipMemAllocationTypeMax cudaMemAllocationTypeMax
#define hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle
//CUarrayMapInfo mappings
typedef CUarrayMapInfo hipArrayMapInfo;
typedef CUarraySparseSubresourceType hipArraySparseSubresourceType;
#define hipArraySparseSubresourceTypeSparseLevel CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
#define hipArraySparseSubresourceTypeMiptail CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
typedef CUmemOperationType hipMemOperationType;
#define hipMemOperationTypeMap CU_MEM_OPERATION_TYPE_MAP
#define hipMemOperationTypeUnmap CU_MEM_OPERATION_TYPE_UNMAP
typedef CUmemHandleType hipMemHandleType;
#define hipMemHandleTypeGeneric CU_MEM_HANDLE_TYPE_GENERIC
// Explicitely declaring hipMemAllocationProp based on CUmemAllocationProp but using CUDA runtime members instead
// Because hipMemAllocationType, hipMemAllocationHandleType & hipMemLocation are defined using CUDA runtime data types & also used by hipMemPoolProps
// Currently there doesn't exist CUDA inbuilt runtime structure corresponding to CUmemAllocationProp
// Need to update this structure accordingly if CUDA updates CUmemAllocationProp
typedef struct hipMemAllocationProp {
    /** Memory allocation type */
    hipMemAllocationType type;
    /** Requested handle type */
    hipMemAllocationHandleType requestedHandleTypes;
    /** Location of allocation */
    hipMemLocation location;
    /**
     * Windows-specific POBJECT_ATTRIBUTES required when
     * ::CU_MEM_HANDLE_TYPE_WIN32 is specified.  This object atributes structure
     * includes security attributes that define
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void *win32HandleMetaData;
    struct {
         /**
         * Allocation hint for requesting compressible memory.
         * On devices that support Compute Data Compression, compressible
         * memory can be used to accelerate accesses to data with unstructured
         * sparsity and other compressible data patterns. Applications are
         * expected to query allocation property of the handle obtained with
         * ::cuMemCreate using ::cuMemGetAllocationPropertiesFromHandle to
         * validate if the obtained allocation is compressible or not. Note that
         * compressed memory may not be mappable on all devices.
         */
         unsigned char compressionType;
         /** RDMA capable */
         unsigned char gpuDirectRDMACapable;
         /** Bitmask indicating intended usage for this allocation */
         unsigned short usage;
         unsigned char reserved[4];
    } allocFlags;
} hipMemAllocationProp;
#endif
/**
 * Stream CallBack struct
 */
#define HIPRT_CB CUDART_CB
typedef void(HIPRT_CB* hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
inline static hipError_t hipInit_cu4oro(unsigned int flags) {
    return hipCUResultTohipError(cuInit(flags));
}

inline static hipError_t hipDeviceReset_cu4oro() { return hipCUDAErrorTohipError(cudaDeviceReset()); }

inline static hipError_t hipGetLastError_cu4oro() { return hipCUDAErrorTohipError(cudaGetLastError()); }

inline static hipError_t hipPeekAtLastError_cu4oro() {
    return hipCUDAErrorTohipError(cudaPeekAtLastError());
}

inline static hipError_t hipMalloc_cu4oro(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMalloc(ptr, size));
}

inline static hipError_t hipMallocPitch_cu4oro(void** ptr, size_t* pitch, size_t width, size_t height) {
    return hipCUDAErrorTohipError(cudaMallocPitch(ptr, pitch, width, height));
}

inline static hipError_t hipMemAllocPitch_cu4oro(hipDeviceptr_t* dptr,size_t* pitch,size_t widthInBytes,size_t height,unsigned int elementSizeBytes){
    return hipCUResultTohipError(cuMemAllocPitch(dptr,pitch,widthInBytes,height,elementSizeBytes));
}

inline static hipError_t hipMalloc3D_cu4oro(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    return hipCUDAErrorTohipError(cudaMalloc3D(pitchedDevPtr, extent));
}

inline static hipError_t hipFree_cu4oro(void* ptr) { return hipCUDAErrorTohipError(cudaFree(ptr)); }

__HIP_DEPRECATED_MSG("use hipHostMalloc_cu4oro instead")
inline static hipError_t hipMallocHost_cu4oro(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMallocHost(ptr, size));
}

__HIP_DEPRECATED_MSG("use hipHostMalloc_cu4oro instead")
inline static hipError_t hipMemAllocHost_cu4oro(void** ptr, size_t size) {
    return hipCUResultTohipError(cuMemAllocHost(ptr, size));
}

__HIP_DEPRECATED_MSG("use hipHostMalloc_cu4oro instead")
inline static hipError_t hipHostAlloc_cu4oro(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

inline static hipError_t hipHostMalloc_cu4oro(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

inline static hipError_t hipMemAdvise_cu4oro(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                                      int device) {
    return hipCUDAErrorTohipError(cudaMemAdvise(dev_ptr, count,
        hipMemoryAdviseTocudaMemoryAdvise(advice), device));
}

inline static hipError_t hipMemPrefetchAsync_cu4oro(const void* dev_ptr, size_t count, int device,
                                             hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemPrefetchAsync(dev_ptr, count, device, stream));
}

inline static hipError_t hipMemRangeGetAttribute_cu4oro(void* data, size_t data_size,
                                                 hipMemRangeAttribute attribute,
                                                 const void* dev_ptr, size_t count) {
    return hipCUDAErrorTohipError(cudaMemRangeGetAttribute(data, data_size,
        hipMemRangeAttributeToCudaMemRangeAttribute(attribute), dev_ptr, count));
}

inline static hipError_t hipMemRangeGetAttributes_cu4oro(void** data, size_t* data_sizes,
                                                  hipMemRangeAttribute* attributes,
                                                  size_t num_attributes, const void* dev_ptr,
                                                  size_t count) {
    return hipCUDAErrorTohipError(cudaMemRangeGetAttributes(data, data_sizes, attributes,
        num_attributes, dev_ptr, count));
}

inline static hipError_t hipStreamAttachMemAsync_cu4oro(hipStream_t stream, hipDeviceptr_t* dev_ptr,
                                                 size_t length __dparm(0),
                                                 unsigned int flags __dparm(hipMemAttachSingle)) {
    return hipCUDAErrorTohipError(cudaStreamAttachMemAsync(stream, dev_ptr, length, flags));
}

inline static hipError_t hipMallocManaged_cu4oro(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMallocManaged(ptr, size, flags));
}

inline static hipError_t hipMallocArray_cu4oro(hipArray** array, const hipChannelFormatDesc* desc,
                                        size_t width, size_t height __dparm(0),
                                        unsigned int flags __dparm(hipArrayDefault)) {
    return hipCUDAErrorTohipError(cudaMallocArray(array, desc, width, height, flags));
}

inline static hipError_t hipMalloc3DArray_cu4oro(hipArray** array, const hipChannelFormatDesc* desc,
                             hipExtent extent, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMalloc3DArray(array, desc, extent, flags));
}

inline static hipError_t hipFreeArray_cu4oro(hipArray* array) {
    return hipCUDAErrorTohipError(cudaFreeArray(array));
}

inline static hipError_t hipMipmappedArrayCreate_cu4oro(hipmipmappedArray* pHandle,
                                                 HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                                 unsigned int numMipmapLevels) {
    return hipCUResultTohipError(cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels));
}

inline static hipError_t hipMipmappedArrayDestroy_cu4oro(hipmipmappedArray hMipmappedArray) {
    return hipCUResultTohipError(cuMipmappedArrayDestroy(hMipmappedArray));
}

inline static hipError_t hipMipmappedArrayGetLevel_cu4oro(hiparray* pLevelArray,
                                                   hipmipmappedArray hMipMappedArray,
                                                   unsigned int level) {
    return hipCUResultTohipError(cuMipmappedArrayGetLevel((CUarray*)pLevelArray, hMipMappedArray, level));
}

inline static hipError_t hipMallocMipmappedArray_cu4oro(hipMipmappedArray_t* pHandle,
                                                 const hipChannelFormatDesc* desc, hipExtent extent,
                                                 unsigned int numLevels, unsigned int flags __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMallocMipmappedArray(pHandle, desc, extent, numLevels, flags));
}

inline static hipError_t hipFreeMipmappedArray_cu4oro(hipMipmappedArray_t hMipmappedArray) {
    return hipCUDAErrorTohipError(cudaFreeMipmappedArray(hMipmappedArray));
}

inline static hipError_t hipGetMipmappedArrayLevel_cu4oro(hipArray_t* pLevelArray,
                                                   hipMipmappedArray_t hMipMappedArray,
                                                   unsigned int level) {
    return hipCUDAErrorTohipError(cudaGetMipmappedArrayLevel(pLevelArray, hMipMappedArray, level));
}

inline static hipError_t hipHostGetDevicePointer_cu4oro(void** devPtr, void* hostPtr, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostGetDevicePointer(devPtr, hostPtr, flags));
}

inline static hipError_t hipHostGetFlags_cu4oro(unsigned int* flagsPtr, void* hostPtr) {
    return hipCUDAErrorTohipError(cudaHostGetFlags(flagsPtr, hostPtr));
}

inline static hipError_t hipHostRegister_cu4oro(void* ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostRegister(ptr, size, flags));
}

inline static hipError_t hipHostUnregister_cu4oro(void* ptr) {
    return hipCUDAErrorTohipError(cudaHostUnregister(ptr));
}

__HIP_DEPRECATED_MSG("use hipHostFree_cu4oro instead")
inline static hipError_t hipFreeHost_cu4oro(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

inline static hipError_t hipHostFree_cu4oro(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

inline static hipError_t hipSetDevice_cu4oro(int device) {
    return hipCUDAErrorTohipError(cudaSetDevice(device));
}

inline static hipError_t hipChooseDevice_cu4oro(int* device, const hipDeviceProp_t* prop) {

    if (prop == NULL) {
      return hipErrorInvalidValue;
    }

    struct cudaDeviceProp cdprop;
    memset(&cdprop, 0x0, sizeof(struct cudaDeviceProp));
    cdprop.major = prop->major;
    cdprop.minor = prop->minor;
    cdprop.totalGlobalMem = prop->totalGlobalMem;
    cdprop.sharedMemPerBlock = prop->sharedMemPerBlock;
    cdprop.regsPerBlock = prop->regsPerBlock;
    cdprop.warpSize = prop->warpSize;
    cdprop.maxThreadsPerBlock = prop->maxThreadsPerBlock;
    cdprop.clockRate = prop->clockRate;
    cdprop.totalConstMem = prop->totalConstMem;
    cdprop.multiProcessorCount = prop->multiProcessorCount;
    cdprop.l2CacheSize = prop->l2CacheSize;
    cdprop.maxThreadsPerMultiProcessor = prop->maxThreadsPerMultiProcessor;
    cdprop.computeMode = prop->computeMode;
    cdprop.canMapHostMemory = prop->canMapHostMemory;
    cdprop.memoryClockRate = prop->memoryClockRate;
    cdprop.memoryBusWidth = prop->memoryBusWidth;
    return hipCUDAErrorTohipError(cudaChooseDevice(device, &cdprop));
}

inline static hipError_t hipMemcpyHtoD_cu4oro(hipDeviceptr_t dst, void* src, size_t size) {
    return hipCUResultTohipError(cuMemcpyHtoD(dst, src, size));
}

inline static hipError_t hipMemcpyDtoH_cu4oro(void* dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoH(dst, src, size));
}

inline static hipError_t hipMemcpyDtoD_cu4oro(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoD(dst, src, size));
}

inline static hipError_t hipMemcpyHtoDAsync_cu4oro(hipDeviceptr_t dst, void* src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyHtoDAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpyDtoHAsync_cu4oro(void* dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoHAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpyDtoDAsync_cu4oro(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoDAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpy_cu4oro(void* dst, const void* src, size_t sizeBytes,
                                   hipMemcpyKind copyKind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy(dst, src, sizeBytes, copyKind));
}


inline static hipError_t hipMemcpyWithStream_cu4oro(void* dst, const void* src, size_t sizeBytes,
                                             hipMemcpyKind copyKind, hipStream_t stream) {
    cudaError_t error = cudaMemcpyAsync(dst, src, sizeBytes, copyKind, stream);

    if (error != cudaSuccess) return hipCUDAErrorTohipError(error);

    return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

inline static hipError_t hipMemcpyAsync_cu4oro(void* dst, const void* src, size_t sizeBytes,
                                        hipMemcpyKind copyKind, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyAsync(dst, src, sizeBytes, copyKind, stream));
}

inline static hipError_t hipMemcpyToSymbol_cu4oro(
    const void* symbol, const void* src, size_t sizeBytes, size_t offset __dparm(0),
    hipMemcpyKind copyType __dparm(hipMemcpyKindToCudaMemcpyKind(hipMemcpyHostToDevice))) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbol(symbol, src, sizeBytes, offset, copyType));
}

inline static hipError_t hipMemcpyToSymbolAsync_cu4oro(const void* symbol, const void* src,
                                                size_t sizeBytes, size_t offset,
                                                hipMemcpyKind copyType,
                                                hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbolAsync(
        symbol, src, sizeBytes, offset, copyType, stream));
}

inline static hipError_t hipMemcpyFromSymbol_cu4oro(
    void* dst, const void* symbolName, size_t sizeBytes, size_t offset __dparm(0),
    hipMemcpyKind kind __dparm(hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost))) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbol(dst, symbolName, sizeBytes, offset, kind));
}

inline static hipError_t hipMemcpyFromSymbolAsync_cu4oro(void* dst, const void* symbolName,
                                                  size_t sizeBytes, size_t offset,
                                                  hipMemcpyKind kind,
                                                  hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbolAsync(
        dst, symbolName, sizeBytes, offset, kind, stream));
}

inline static hipError_t hipGetSymbolAddress_cu4oro(void** devPtr, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolAddress(devPtr, symbolName));
}

inline static hipError_t hipGetSymbolSize_cu4oro(size_t* size, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolSize(size, symbolName));
}

inline static hipError_t hipMemcpy2D_cu4oro(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

inline static hipError_t hipMemcpyParam2D_cu4oro(const hip_Memcpy2D* pCopy) {
  return hipCUResultTohipError(cuMemcpy2D(pCopy));
}

inline static hipError_t hipMemcpyParam2DAsync_cu4oro(const hip_Memcpy2D* pCopy, hipStream_t stream __dparm(0)) {
  return hipCUResultTohipError(cuMemcpy2DAsync(pCopy, stream));
}

inline static hipError_t hipMemcpy3D_cu4oro(const struct hipMemcpy3DParms *p) {
    return hipCUDAErrorTohipError(cudaMemcpy3D(p));
}

inline static hipError_t hipMemcpy3DAsync_cu4oro(const struct hipMemcpy3DParms *p, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy3DAsync(p, stream));
}

inline static hipError_t hipDrvMemcpy3D_cu4oro(const HIP_MEMCPY3D* pCopy) {
    return hipCUResultTohipError(cuMemcpy3D(pCopy));
}

inline static hipError_t hipDrvMemcpy3DAsync_cu4oro(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpy3DAsync(pCopy, stream));
}

inline static hipError_t hipMemcpy2DAsync_cu4oro(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, hipMemcpyKind kind,
                                          hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                                    kind, stream));
}

inline static hipError_t hipMemcpy2DFromArray_cu4oro(void* dst, size_t dpitch, hipArray* src,
                                              size_t wOffset, size_t hOffset, size_t width,
                                              size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width,
                                                        height,
                                                        kind));
}

inline static hipError_t hipMemcpy2DFromArrayAsync_cu4oro(void* dst, size_t dpitch, hipArray* src,
                                                   size_t wOffset, size_t hOffset, size_t width,
                                                   size_t height, hipMemcpyKind kind,
                                                   hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset,
                                                             width, height,
                                                             kind,
                                                             stream));
}

inline static hipError_t hipMemcpy2DToArray_cu4oro(hipArray* dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t spitch, size_t width,
                                            size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width,
                                                      height, kind));
}

inline static hipError_t hipMemcpy2DToArrayAsync_cu4oro(hipArray* dst, size_t wOffset, size_t hOffset,
                                                 const void* src, size_t spitch, size_t width,
                                                 size_t height, hipMemcpyKind kind,
                                                 hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch,
                                                           width, height,
                                                           kind,
                                                           stream));
}

__HIP_DEPRECATED inline static hipError_t hipMemcpyToArray_cu4oro(hipArray* dst, size_t wOffset,
                                                           size_t hOffset, const void* src,
                                                           size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind));
}

__HIP_DEPRECATED inline static hipError_t hipMemcpyFromArray_cu4oro(void* dst, hipArray_const_t srcArray,
                                                             size_t wOffset, size_t hOffset,
                                                             size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpyFromArray(dst, srcArray, wOffset, hOffset, count,
                                                      kind));
}

inline static hipError_t hipMemcpyAtoH_cu4oro(void* dst, hipArray* srcArray, size_t srcOffset,
                                       size_t count) {
    return hipCUResultTohipError(cuMemcpyAtoH(dst, (CUarray)srcArray, srcOffset, count));
}

inline static hipError_t hipMemcpyHtoA_cu4oro(hipArray* dstArray, size_t dstOffset, const void* srcHost,
                                       size_t count) {
    return hipCUResultTohipError(cuMemcpyHtoA((CUarray)dstArray, dstOffset, srcHost, count));
}

inline static hipError_t hipDeviceSynchronize_cu4oro() {
    return hipCUDAErrorTohipError(cudaDeviceSynchronize());
}

inline static hipError_t hipDeviceGetCacheConfig_cu4oro(hipFuncCache_t* pCacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceGetCacheConfig(pCacheConfig));
}

inline static hipError_t hipFuncSetAttribute_cu4oro(const void* func, hipFuncAttribute attr, int value) {
    return hipCUDAErrorTohipError(cudaFuncSetAttribute(func, attr, value));
}

inline static hipError_t hipDeviceSetCacheConfig_cu4oro(hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceSetCacheConfig(cacheConfig));
}

inline static hipError_t hipFuncSetSharedMemConfig_cu4oro(const void* func, hipSharedMemConfig config) {
    return hipCUDAErrorTohipError(cudaFuncSetSharedMemConfig(func, config));
}

inline static const char* hipGetErrorString_cu4oro(hipError_t error) {
    return cudaGetErrorString(hipErrorToCudaError(error));
}

inline static const char* hipGetErrorName_cu4oro(hipError_t error) {
    return cudaGetErrorName(hipErrorToCudaError(error));
}

inline static hipError_t hipDrvGetErrorString_cu4oro(hipError_t error, const char** errorString) {
    CUresult err = hipErrorToCUResult(error);
    if( err == CUDA_ERROR_UNKNOWN ) {
       return hipCUResultTohipError(cuGetErrorString((CUresult)error, errorString));
    } else {
       return hipCUResultTohipError(cuGetErrorString(err, errorString));
    }
}

inline static hipError_t hipDrvGetErrorName_cu4oro(hipError_t error, const char** errorString) {
    CUresult err = hipErrorToCUResult(error);
    if( err == CUDA_ERROR_UNKNOWN ) {
       return hipCUResultTohipError(cuGetErrorName((CUresult)error, errorString));
    } else {
       return hipCUResultTohipError(cuGetErrorName(err, errorString));
    }
}

inline static hipError_t hipGetDeviceCount_cu4oro(int* count) {
    return hipCUDAErrorTohipError(cudaGetDeviceCount(count));
}

inline static hipError_t hipGetDevice_cu4oro(int* device) {
    return hipCUDAErrorTohipError(cudaGetDevice(device));
}

inline static hipError_t hipIpcCloseMemHandle_cu4oro(void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcCloseMemHandle(devPtr));
}

inline static hipError_t hipIpcGetEventHandle_cu4oro(hipIpcEventHandle_t* handle, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaIpcGetEventHandle(handle, event));
}

inline static hipError_t hipIpcGetMemHandle_cu4oro(hipIpcMemHandle_t* handle, void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcGetMemHandle(handle, devPtr));
}

inline static hipError_t hipIpcOpenEventHandle_cu4oro(hipEvent_t* event, hipIpcEventHandle_t handle) {
    return hipCUDAErrorTohipError(cudaIpcOpenEventHandle(event, handle));
}

inline static hipError_t hipIpcOpenMemHandle_cu4oro(void** devPtr, hipIpcMemHandle_t handle,
                                             unsigned int flags) {
    return hipCUDAErrorTohipError(cudaIpcOpenMemHandle(devPtr, handle, flags));
}

inline static hipError_t hipMemset_cu4oro(void* devPtr, int value, size_t count) {
    return hipCUDAErrorTohipError(cudaMemset(devPtr, value, count));
}

inline static hipError_t hipMemsetD32_cu4oro(hipDeviceptr_t devPtr, int value, size_t count) {
    return hipCUResultTohipError(cuMemsetD32(devPtr, value, count));
}

inline static hipError_t hipMemsetAsync_cu4oro(void* devPtr, int value, size_t count,
                                        hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemsetAsync(devPtr, value, count, stream));
}

inline static hipError_t hipMemsetD32Async_cu4oro(hipDeviceptr_t devPtr, int value, size_t count,
                                           hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD32Async(devPtr, value, count, stream));
}

inline static hipError_t hipMemsetD8_cu4oro(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD8(dest, value, sizeBytes));
}

inline static hipError_t hipMemsetD8Async_cu4oro(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes,
                                          hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD8Async(dest, value, sizeBytes, stream));
}

inline static hipError_t hipMemsetD16_cu4oro(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD16(dest, value, sizeBytes));
}

inline static hipError_t hipMemsetD16Async_cu4oro(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes,
                                           hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD16Async(dest, value, sizeBytes, stream));
}

inline static hipError_t hipMemset2D_cu4oro(void* dst, size_t pitch, int value, size_t width, size_t height) {
    return hipCUDAErrorTohipError(cudaMemset2D(dst, pitch, value, width, height));
}

inline static hipError_t hipMemset2DAsync_cu4oro(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemset2DAsync(dst, pitch, value, width, height, stream));
}

inline static hipError_t hipMemset3D_cu4oro(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ){
    return hipCUDAErrorTohipError(cudaMemset3D(pitchedDevPtr, value, extent));
}

inline static hipError_t hipMemset3DAsync_cu4oro(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent, hipStream_t stream __dparm(0) ){
    return hipCUDAErrorTohipError(cudaMemset3DAsync(pitchedDevPtr, value, extent, stream));
}

inline static hipError_t hipGetDeviceProperties_cu4oro(hipDeviceProp_t* p_prop, int device) {

    if (p_prop == NULL) {
      return hipErrorInvalidValue;
    }

    struct cudaDeviceProp cdprop;
    cudaError_t cerror;
    cerror = cudaGetDeviceProperties(&cdprop, device);

    strncpy(p_prop->name, cdprop.name, 256);
    p_prop->totalGlobalMem = cdprop.totalGlobalMem;
    p_prop->sharedMemPerBlock = cdprop.sharedMemPerBlock;
    p_prop->regsPerBlock = cdprop.regsPerBlock;
    p_prop->warpSize = cdprop.warpSize;
    p_prop->maxThreadsPerBlock = cdprop.maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        p_prop->maxThreadsDim[i] = cdprop.maxThreadsDim[i];
        p_prop->maxGridSize[i] = cdprop.maxGridSize[i];
    }
    p_prop->clockRate = cdprop.clockRate;
    p_prop->memoryClockRate = cdprop.memoryClockRate;
    p_prop->memoryBusWidth = cdprop.memoryBusWidth;
    p_prop->totalConstMem = cdprop.totalConstMem;
    p_prop->major = cdprop.major;
    p_prop->minor = cdprop.minor;
    p_prop->multiProcessorCount = cdprop.multiProcessorCount;
    p_prop->l2CacheSize = cdprop.l2CacheSize;
    p_prop->maxThreadsPerMultiProcessor = cdprop.maxThreadsPerMultiProcessor;
    p_prop->computeMode = cdprop.computeMode;
    p_prop->clockInstructionRate = cdprop.clockRate; // Same as clock-rate:

    int ccVers = p_prop->major * 100 + p_prop->minor * 10;
    p_prop->arch.hasGlobalInt32Atomics = (ccVers >= 110);
    p_prop->arch.hasGlobalFloatAtomicExch = (ccVers >= 110);
    p_prop->arch.hasSharedInt32Atomics = (ccVers >= 120);
    p_prop->arch.hasSharedFloatAtomicExch = (ccVers >= 120);
    p_prop->arch.hasFloatAtomicAdd = (ccVers >= 200);
    p_prop->arch.hasGlobalInt64Atomics = (ccVers >= 120);
    p_prop->arch.hasSharedInt64Atomics = (ccVers >= 110);
    p_prop->arch.hasDoubles = (ccVers >= 130);
    p_prop->arch.hasWarpVote = (ccVers >= 120);
    p_prop->arch.hasWarpBallot = (ccVers >= 200);
    p_prop->arch.hasWarpShuffle = (ccVers >= 300);
    p_prop->arch.hasFunnelShift = (ccVers >= 350);
    p_prop->arch.hasThreadFenceSystem = (ccVers >= 200);
    p_prop->arch.hasSyncThreadsExt = (ccVers >= 200);
    p_prop->arch.hasSurfaceFuncs = (ccVers >= 200);
    p_prop->arch.has3dGrid = (ccVers >= 200);
    p_prop->arch.hasDynamicParallelism = (ccVers >= 350);

    p_prop->concurrentKernels = cdprop.concurrentKernels;
    p_prop->pciDomainID = cdprop.pciDomainID;
    p_prop->pciBusID = cdprop.pciBusID;
    p_prop->pciDeviceID = cdprop.pciDeviceID;
    p_prop->maxSharedMemoryPerMultiProcessor = cdprop.sharedMemPerMultiprocessor;
    p_prop->isMultiGpuBoard = cdprop.isMultiGpuBoard;
    p_prop->canMapHostMemory = cdprop.canMapHostMemory;
    p_prop->gcnArch = 0; // Not a GCN arch
    p_prop->integrated = cdprop.integrated;
    p_prop->cooperativeLaunch = cdprop.cooperativeLaunch;
    p_prop->cooperativeMultiDeviceLaunch = cdprop.cooperativeMultiDeviceLaunch;
    p_prop->cooperativeMultiDeviceUnmatchedFunc = 0;
    p_prop->cooperativeMultiDeviceUnmatchedGridDim = 0;
    p_prop->cooperativeMultiDeviceUnmatchedBlockDim = 0;
    p_prop->cooperativeMultiDeviceUnmatchedSharedMem = 0;

    p_prop->maxTexture1D    = cdprop.maxTexture1D;
    p_prop->maxTexture2D[0] = cdprop.maxTexture2D[0];
    p_prop->maxTexture2D[1] = cdprop.maxTexture2D[1];
    p_prop->maxTexture3D[0] = cdprop.maxTexture3D[0];
    p_prop->maxTexture3D[1] = cdprop.maxTexture3D[1];
    p_prop->maxTexture3D[2] = cdprop.maxTexture3D[2];

    p_prop->memPitch                 = cdprop.memPitch;
    p_prop->textureAlignment         = cdprop.textureAlignment;
    p_prop->texturePitchAlignment    = cdprop.texturePitchAlignment;
    p_prop->kernelExecTimeoutEnabled = cdprop.kernelExecTimeoutEnabled;
    p_prop->ECCEnabled               = cdprop.ECCEnabled;
    p_prop->tccDriver                = cdprop.tccDriver;

    return hipCUDAErrorTohipError(cerror);
}

inline static hipError_t hipDeviceGetAttribute_cu4oro(int* pi, hipDeviceAttribute_t attr, int device) {
    enum cudaDeviceAttr cdattr;
    cudaError_t cerror;

    switch (attr) {
        case hipDeviceAttributeMaxThreadsPerBlock:
            cdattr = cudaDevAttrMaxThreadsPerBlock;
            break;
        case hipDeviceAttributeMaxBlockDimX:
            cdattr = cudaDevAttrMaxBlockDimX;
            break;
        case hipDeviceAttributeMaxBlockDimY:
            cdattr = cudaDevAttrMaxBlockDimY;
            break;
        case hipDeviceAttributeMaxBlockDimZ:
            cdattr = cudaDevAttrMaxBlockDimZ;
            break;
        case hipDeviceAttributeMaxGridDimX:
            cdattr = cudaDevAttrMaxGridDimX;
            break;
        case hipDeviceAttributeMaxGridDimY:
            cdattr = cudaDevAttrMaxGridDimY;
            break;
        case hipDeviceAttributeMaxGridDimZ:
            cdattr = cudaDevAttrMaxGridDimZ;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            cdattr = cudaDevAttrMaxSharedMemoryPerBlock;
            break;
        case hipDeviceAttributeTotalConstantMemory:
            cdattr = cudaDevAttrTotalConstantMemory;
            break;
        case hipDeviceAttributeWarpSize:
            cdattr = cudaDevAttrWarpSize;
            break;
        case hipDeviceAttributeMaxRegistersPerBlock:
            cdattr = cudaDevAttrMaxRegistersPerBlock;
            break;
        case hipDeviceAttributeClockRate:
            cdattr = cudaDevAttrClockRate;
            break;
        case hipDeviceAttributeMemoryClockRate:
            cdattr = cudaDevAttrMemoryClockRate;
            break;
        case hipDeviceAttributeMemoryBusWidth:
            cdattr = cudaDevAttrGlobalMemoryBusWidth;
            break;
        case hipDeviceAttributeMultiprocessorCount:
            cdattr = cudaDevAttrMultiProcessorCount;
            break;
        case hipDeviceAttributeComputeMode:
            cdattr = cudaDevAttrComputeMode;
            break;
        case hipDeviceAttributeL2CacheSize:
            cdattr = cudaDevAttrL2CacheSize;
            break;
        case hipDeviceAttributeMaxThreadsPerMultiProcessor:
            cdattr = cudaDevAttrMaxThreadsPerMultiProcessor;
            break;
        case hipDeviceAttributeComputeCapabilityMajor:
            cdattr = cudaDevAttrComputeCapabilityMajor;
            break;
        case hipDeviceAttributeComputeCapabilityMinor:
            cdattr = cudaDevAttrComputeCapabilityMinor;
            break;
        case hipDeviceAttributeConcurrentKernels:
            cdattr = cudaDevAttrConcurrentKernels;
            break;
        case hipDeviceAttributePciBusId:
            cdattr = cudaDevAttrPciBusId;
            break;
        case hipDeviceAttributePciDeviceId:
            cdattr = cudaDevAttrPciDeviceId;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
            cdattr = cudaDevAttrMaxSharedMemoryPerMultiprocessor;
            break;
        case hipDeviceAttributeIsMultiGpuBoard:
            cdattr = cudaDevAttrIsMultiGpuBoard;
            break;
        case hipDeviceAttributeIntegrated:
            cdattr = cudaDevAttrIntegrated;
            break;
        case hipDeviceAttributeMaxTexture1DWidth:
            cdattr = cudaDevAttrMaxTexture1DWidth;
            break;
        case hipDeviceAttributeMaxTexture2DWidth:
            cdattr = cudaDevAttrMaxTexture2DWidth;
            break;
        case hipDeviceAttributeMaxTexture2DHeight:
            cdattr = cudaDevAttrMaxTexture2DHeight;
            break;
        case hipDeviceAttributeMaxTexture3DWidth:
            cdattr = cudaDevAttrMaxTexture3DWidth;
            break;
        case hipDeviceAttributeMaxTexture3DHeight:
            cdattr = cudaDevAttrMaxTexture3DHeight;
            break;
        case hipDeviceAttributeMaxTexture3DDepth:
            cdattr = cudaDevAttrMaxTexture3DDepth;
            break;
        case hipDeviceAttributeMaxPitch:
            cdattr = cudaDevAttrMaxPitch;
            break;
        case hipDeviceAttributeTextureAlignment:
            cdattr = cudaDevAttrTextureAlignment;
            break;
        case hipDeviceAttributeTexturePitchAlignment:
            cdattr = cudaDevAttrTexturePitchAlignment;
            break;
        case hipDeviceAttributeKernelExecTimeout:
            cdattr = cudaDevAttrKernelExecTimeout;
            break;
        case hipDeviceAttributeCanMapHostMemory:
            cdattr = cudaDevAttrCanMapHostMemory;
            break;
        case hipDeviceAttributeEccEnabled:
            cdattr = cudaDevAttrEccEnabled;
            break;
        case hipDeviceAttributeCooperativeLaunch:
            cdattr = cudaDevAttrCooperativeLaunch;
            break;
        case hipDeviceAttributeCooperativeMultiDeviceLaunch:
            cdattr = cudaDevAttrCooperativeMultiDeviceLaunch;
            break;
        case hipDeviceAttributeConcurrentManagedAccess:
            cdattr = cudaDevAttrConcurrentManagedAccess;
            break;
        case hipDeviceAttributeManagedMemory:
            cdattr = cudaDevAttrManagedMemory;
            break;
        case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
            cdattr = cudaDevAttrPageableMemoryAccessUsesHostPageTables;
            break;
        case hipDeviceAttributePageableMemoryAccess:
            cdattr = cudaDevAttrPageableMemoryAccess;
            break;
        case hipDeviceAttributeDirectManagedMemAccessFromHost:
            cdattr = cudaDevAttrDirectManagedMemAccessFromHost;
            break;
        case hipDeviceAttributeGlobalL1CacheSupported:
            cdattr = cudaDevAttrGlobalL1CacheSupported;
            break;
        case hipDeviceAttributeMaxBlocksPerMultiProcessor:
            cdattr = cudaDevAttrMaxBlocksPerMultiprocessor;
            break;
        case hipDeviceAttributeMultiGpuBoardGroupID:
            cdattr = cudaDevAttrMultiGpuBoardGroupID;
            break;
        case hipDeviceAttributeReservedSharedMemPerBlock:
            cdattr = cudaDevAttrReservedSharedMemoryPerBlock;
            break;
        case hipDeviceAttributeSingleToDoublePrecisionPerfRatio:
            cdattr = cudaDevAttrSingleToDoublePrecisionPerfRatio;
            break;
        case hipDeviceAttributeStreamPrioritiesSupported:
            cdattr = cudaDevAttrStreamPrioritiesSupported;
            break;
        case hipDeviceAttributeSurfaceAlignment:
            cdattr = cudaDevAttrSurfaceAlignment;
            break;
        case hipDeviceAttributeTccDriver:
            cdattr = cudaDevAttrTccDriver;
            break;
        case hipDeviceAttributeUnifiedAddressing:
            cdattr = cudaDevAttrUnifiedAddressing;
            break;
#if CUDA_VERSION >= CUDA_11020
        case hipDeviceAttributeMemoryPoolsSupported:
            cdattr = cudaDevAttrMemoryPoolsSupported;
            break;
#endif // CUDA_VERSION >= CUDA_11020
        case hipDeviceAttributeVirtualMemoryManagementSupported:
            return hipCUResultTohipError(cuDeviceGetAttribute(pi,
                                                              CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                                              device));
        case hipDeviceAttributeAccessPolicyMaxWindowSize:
            cdattr = cudaDevAttrMaxAccessPolicyWindowSize;
            break;
        case hipDeviceAttributeAsyncEngineCount:
            cdattr = cudaDevAttrAsyncEngineCount;
            break;
        case hipDeviceAttributeCanUseHostPointerForRegisteredMem:
            cdattr = cudaDevAttrCanUseHostPointerForRegisteredMem;
            break;
        case hipDeviceAttributeComputePreemptionSupported:
            cdattr = cudaDevAttrComputePreemptionSupported;
            break;
        case hipDeviceAttributeHostNativeAtomicSupported:
            cdattr = cudaDevAttrHostNativeAtomicSupported;
            break;
        default:
            return hipCUDAErrorTohipError(cudaErrorInvalidValue);
    }
    cerror = cudaDeviceGetAttribute(pi, cdattr, device);
    return hipCUDAErrorTohipError(cerror);
}
#if CUDA_VERSION >= CUDA_10020
inline static CUmemAllocationProp hipMemAllocationPropToCUmemAllocationProp(const hipMemAllocationProp* prop) {
    CUmemAllocationProp cuProp;
    cuProp.type = (CUmemAllocationType)prop->type;
    cuProp.requestedHandleTypes = (CUmemAllocationHandleType)prop->requestedHandleTypes;
    cuProp.location.type = (CUmemLocationType)prop->location.type;
    cuProp.location.id = prop->location.id;
    cuProp.win32HandleMetaData = prop->win32HandleMetaData;
    cuProp.allocFlags.compressionType = prop->allocFlags.compressionType;
    cuProp.allocFlags.gpuDirectRDMACapable = prop->allocFlags.gpuDirectRDMACapable;
    cuProp.allocFlags.usage = prop->allocFlags.usage;
    cuProp.allocFlags.reserved[0] = prop->allocFlags.reserved[0];
    cuProp.allocFlags.reserved[1] = prop->allocFlags.reserved[1];
    cuProp.allocFlags.reserved[2] = prop->allocFlags.reserved[2];
    cuProp.allocFlags.reserved[3] = prop->allocFlags.reserved[3];
    return cuProp;
}
inline static CUmemLocation hipMemLocationToCUmemLocation(const hipMemLocation* loc) {
    CUmemLocation cuLoc;
    cuLoc.id = loc->id;
    cuLoc.type = (CUmemLocationType)loc->type;
    return cuLoc;
}
inline static CUmemAccessDesc hipMemAccessDescToCUmemAccessDesc(const hipMemAccessDesc* desc) {
    CUmemAccessDesc cuDesc;
    cuDesc.flags = (CUmemAccess_flags)desc->flags;
    cuDesc.location.id = (desc->location).id;
    cuDesc.location.type = (CUmemLocationType)((desc->location).type);
    return cuDesc;
}
inline static hipError_t hipMemGetAllocationGranularity_cu4oro(size_t* granularity,
                                                        const hipMemAllocationProp* prop,
                                                        hipMemAllocationGranularity_flags option) {
    CUmemAllocationProp cuProp = hipMemAllocationPropToCUmemAllocationProp(prop);
    return hipCUResultTohipError(cuMemGetAllocationGranularity(granularity, &cuProp, option));
}
inline static hipError_t hipMemCreate_cu4oro(hipMemGenericAllocationHandle_t* handle,
                                      size_t size,
                                      const hipMemAllocationProp* prop,
                                      unsigned long long flags) {
    CUmemAllocationProp cuProp = hipMemAllocationPropToCUmemAllocationProp(prop);
    return hipCUResultTohipError(cuMemCreate(handle, size, &cuProp, flags));
}
inline static hipError_t hipMemRelease_cu4oro(hipMemGenericAllocationHandle_t handle) {
    return hipCUResultTohipError(cuMemRelease(handle));
}
inline static hipError_t hipMemAddressFree_cu4oro(hipDeviceptr_t ptr, size_t size) {
    return hipCUResultTohipError(cuMemAddressFree(ptr, size));
}
inline static hipError_t hipMemAddressReserve_cu4oro(hipDeviceptr_t* ptr,
                                              size_t size,
                                              size_t alignment,
                                              hipDeviceptr_t addr,
                                              unsigned long long flags) {
    return hipCUResultTohipError(cuMemAddressReserve(ptr, size, alignment, addr, flags));
}
inline static hipError_t hipMemExportToShareableHandle_cu4oro(void* shareableHandle,
                                                       hipMemGenericAllocationHandle_t handle,
                                                       hipMemAllocationHandleType handleType,
                                                       unsigned long long flags) {
    return hipCUResultTohipError(cuMemExportToShareableHandle(shareableHandle, handle, (CUmemAllocationHandleType)handleType, flags));
}
inline static hipError_t hipMemGetAccess_cu4oro(unsigned long long* flags,
                                         const hipMemLocation* location,
                                         hipDeviceptr_t ptr) {
    CUmemLocation loc = hipMemLocationToCUmemLocation(location);
    return hipCUResultTohipError(cuMemGetAccess(flags, &loc, ptr));
}
inline static hipError_t hipMemGetAllocationPropertiesFromHandle_cu4oro(hipMemAllocationProp* prop,
                                                                 hipMemGenericAllocationHandle_t handle) {
    CUmemAllocationProp cuProp = hipMemAllocationPropToCUmemAllocationProp(prop);
    return hipCUResultTohipError(cuMemGetAllocationPropertiesFromHandle(&cuProp, handle));
}
inline static hipError_t hipMemImportFromShareableHandle_cu4oro(hipMemGenericAllocationHandle_t* handle,
                                                         void* osHandle,
                                                         hipMemAllocationHandleType shHandleType) {
    return hipCUResultTohipError(cuMemImportFromShareableHandle(handle, osHandle, (CUmemAllocationHandleType)shHandleType));
}
inline static hipError_t hipMemMap_cu4oro(hipDeviceptr_t ptr, size_t size, size_t offset,
                                   hipMemGenericAllocationHandle_t handle,
                                   unsigned long long flags) {
    return hipCUResultTohipError(cuMemMap(ptr, size, offset, handle, flags));
}
inline static hipError_t hipMemMapArrayAsync_cu4oro(hipArrayMapInfo* mapInfoList,
                                             unsigned int  count,
                                             hipStream_t stream) {
    return hipCUResultTohipError(cuMemMapArrayAsync(mapInfoList, count, stream));
}
inline static hipError_t hipMemRetainAllocationHandle_cu4oro(hipMemGenericAllocationHandle_t* handle,
                                                      void* addr) {
    return hipCUResultTohipError(cuMemRetainAllocationHandle(handle, addr));
}
inline static hipError_t hipMemSetAccess_cu4oro(hipDeviceptr_t ptr, size_t size,
                                         const hipMemAccessDesc* desc,
                                         size_t count) {
    CUmemAccessDesc cuDesc = hipMemAccessDescToCUmemAccessDesc(desc);
    return hipCUResultTohipError(cuMemSetAccess(ptr, size, &cuDesc, count));
}
inline static hipError_t hipMemUnmap_cu4oro(hipDeviceptr_t ptr, size_t size) {
    return hipCUResultTohipError(cuMemUnmap(ptr, size));
}
#endif // CUDA_VERSION >= CUDA_10020

inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor_cu4oro(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                              blockSize, dynamicSMemSize));
}

inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cu4oro(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize,
                                                                      unsigned int flags) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func,
                                                      blockSize, dynamicSMemSize, flags));
}

inline static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_cu4oro(int* numBlocks,
                                                                 hipFunction_t f,
                                                                 int  blockSize,
                                                                 size_t dynamicSMemSize ){
    return hipCUResultTohipError(cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f,
                                                                   blockSize, dynamicSMemSize));
}

inline static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cu4oro(int* numBlocks,
                                                                          hipFunction_t f,
                                                                          int  blockSize,
                                                                          size_t dynamicSMemSize,
                                                                          unsigned int  flags ) {
    return hipCUResultTohipError(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,
                                                                blockSize, dynamicSMemSize, flags));
}

//TODO - Match CUoccupancyB2DSize
inline static hipError_t hipModuleOccupancyMaxPotentialBlockSize_cu4oro(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit){
    return hipCUResultTohipError(cuOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, NULL,
                                 dynSharedMemPerBlk, blockSizeLimit));
}

//TODO - Match CUoccupancyB2DSize
inline static hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags_cu4oro(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int  flags){
    return hipCUResultTohipError(cuOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, f, NULL,
                                 dynSharedMemPerBlk, blockSizeLimit, flags));
}

inline static hipError_t hipPointerGetAttributes_cu4oro(hipPointerAttribute_t* attributes, const void* ptr) {
    struct cudaPointerAttributes cPA;
    hipError_t err = hipCUDAErrorTohipError(cudaPointerGetAttributes(&cPA, ptr));
    if (err == hipSuccess) {
#if (CUDART_VERSION >= 11000)
        enum cudaMemoryType memType = cPA.type;
#else
        unsigned memType = cPA.memoryType; // No auto because cuda 10.2 doesnt force c++11
#endif
        switch (memType) {
            case cudaMemoryTypeDevice:
                attributes->type = hipMemoryTypeDevice;
                break;
            case cudaMemoryTypeHost:
                attributes->type = hipMemoryTypeHost;
                break;
            case cudaMemoryTypeManaged:
                attributes->type = hipMemoryTypeManaged;
                break;
            default:
                return hipErrorInvalidValue;
        }
        attributes->device = cPA.device;
        attributes->devicePointer = cPA.devicePointer;
        attributes->hostPointer = cPA.hostPointer;
        attributes->isManaged = 0;
        attributes->allocationFlags = 0;
    }
    return err;
}

inline static hipError_t hipPointerGetAttribute_cu4oro(void* data, hipPointer_attribute attribute,
                                                hipDeviceptr_t ptr) {
    return hipCUResultTohipError(cuPointerGetAttribute(data, attribute, ptr));
}

inline static hipError_t hipDrvPointerGetAttributes_cu4oro(unsigned int numAttributes,
                                                    hipPointer_attribute* attributes,
                                                    void** data, hipDeviceptr_t ptr) {
    return hipCUResultTohipError(cuPointerGetAttributes(numAttributes, attributes, data, ptr));
}

inline static hipError_t hipMemGetInfo_cu4oro(size_t* free, size_t* total) {
    return hipCUDAErrorTohipError(cudaMemGetInfo(free, total));
}

inline static hipError_t hipEventCreate_cu4oro(hipEvent_t* event) {
    return hipCUDAErrorTohipError(cudaEventCreate(event));
}

inline static hipError_t hipEventRecord_cu4oro(hipEvent_t event, hipStream_t stream __dparm(NULL)) {
    return hipCUDAErrorTohipError(cudaEventRecord(event, stream));
}

inline static hipError_t hipEventSynchronize_cu4oro(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventSynchronize(event));
}

inline static hipError_t hipEventElapsedTime_cu4oro(float* ms, hipEvent_t start, hipEvent_t stop) {
    return hipCUDAErrorTohipError(cudaEventElapsedTime(ms, start, stop));
}

inline static hipError_t hipEventDestroy_cu4oro(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventDestroy(event));
}

inline static hipError_t hipStreamCreateWithFlags_cu4oro(hipStream_t* stream, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithFlags(stream, flags));
}

inline static hipError_t hipStreamCreateWithPriority_cu4oro(hipStream_t* stream, unsigned int flags, int priority) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithPriority(stream, flags, priority));
}

inline static hipError_t hipDeviceGetStreamPriorityRange_cu4oro(int* leastPriority, int* greatestPriority) {
    return hipCUDAErrorTohipError(cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
}

inline static hipError_t hipStreamCreate_cu4oro(hipStream_t* stream) {
    return hipCUDAErrorTohipError(cudaStreamCreate(stream));
}

inline static hipError_t hipStreamSynchronize_cu4oro(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

inline static hipError_t hipStreamDestroy_cu4oro(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamDestroy(stream));
}

inline static hipError_t hipStreamGetFlags_cu4oro(hipStream_t stream, unsigned int *flags) {
    return hipCUDAErrorTohipError(cudaStreamGetFlags(stream, flags));
}

inline static hipError_t hipStreamGetPriority_cu4oro(hipStream_t stream, int *priority) {
    return hipCUDAErrorTohipError(cudaStreamGetPriority(stream, priority));
}

inline static hipError_t hipStreamWaitEvent_cu4oro(hipStream_t stream, hipEvent_t event,
                                            unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamWaitEvent(stream, event, flags));
}

inline static hipError_t hipStreamQuery_cu4oro(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamQuery(stream));
}

inline static hipError_t hipStreamAddCallback_cu4oro(hipStream_t stream, hipStreamCallback_t callback,
                                              void* userData, unsigned int flags) {
    return hipCUDAErrorTohipError(
        cudaStreamAddCallback(stream, (cudaStreamCallback_t)callback, userData, flags));
}

inline static hipError_t hipStreamGetDevice_cu4oro(hipStream_t stream, hipDevice_t* device) {
    hipCtx_t context;
    hipError_t err = hipCUResultTohipError(cuStreamGetCtx(stream, &context));
    if (err != hipSuccess) return err;

    err = hipCUResultTohipError(cuCtxPushCurrent(context));
    if (err != hipSuccess) return err;

    err = hipCUResultTohipError(cuCtxGetDevice(device));
    if (err != hipSuccess) return err;

    return hipCUResultTohipError(cuCtxPopCurrent(&context));
}

inline static hipError_t hipDriverGetVersion_cu4oro(int* driverVersion) {
    return hipCUDAErrorTohipError(cudaDriverGetVersion(driverVersion));
}

inline static hipError_t hipRuntimeGetVersion_cu4oro(int* runtimeVersion) {
    return hipCUDAErrorTohipError(cudaRuntimeGetVersion(runtimeVersion));
}

inline static hipError_t hipDeviceCanAccessPeer_cu4oro(int* canAccessPeer, int device, int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice));
}

inline static hipError_t hipDeviceDisablePeerAccess_cu4oro(int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceDisablePeerAccess(peerDevice));
}

inline static hipError_t hipDeviceEnablePeerAccess_cu4oro(int peerDevice, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaDeviceEnablePeerAccess(peerDevice, flags));
}

inline static hipError_t hipCtxDisablePeerAccess_cu4oro(hipCtx_t peerCtx) {
    return hipCUResultTohipError(cuCtxDisablePeerAccess(peerCtx));
}

inline static hipError_t hipCtxEnablePeerAccess_cu4oro(hipCtx_t peerCtx, unsigned int flags) {
    return hipCUResultTohipError(cuCtxEnablePeerAccess(peerCtx, flags));
}

inline static hipError_t hipDevicePrimaryCtxGetState_cu4oro(hipDevice_t dev, unsigned int* flags,
                                                     int* active) {
    return hipCUResultTohipError(cuDevicePrimaryCtxGetState(dev, flags, active));
}

inline static hipError_t hipDevicePrimaryCtxRelease_cu4oro(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRelease(dev));
}

inline static hipError_t hipDevicePrimaryCtxRetain_cu4oro(hipCtx_t* pctx, hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRetain(pctx, dev));
}

inline static hipError_t hipDevicePrimaryCtxReset_cu4oro(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxReset(dev));
}

inline static hipError_t hipDevicePrimaryCtxSetFlags_cu4oro(hipDevice_t dev, unsigned int flags) {
    return hipCUResultTohipError(cuDevicePrimaryCtxSetFlags(dev, flags));
}

inline static hipError_t hipMemGetAddressRange_cu4oro(hipDeviceptr_t* pbase, size_t* psize,
                                               hipDeviceptr_t dptr) {
    return hipCUResultTohipError(cuMemGetAddressRange(pbase, psize, dptr));
}

inline static hipError_t hipMemcpyPeer_cu4oro(void* dst, int dstDevice, const void* src, int srcDevice,
                                       size_t count) {
    return hipCUDAErrorTohipError(cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count));
}

inline static hipError_t hipMemcpyPeerAsync_cu4oro(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count,
                                            hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream));
}

// Profile APIs:
inline static hipError_t hipProfilerStart_cu4oro() { return hipCUDAErrorTohipError(cudaProfilerStart()); }

inline static hipError_t hipProfilerStop_cu4oro() { return hipCUDAErrorTohipError(cudaProfilerStop()); }

inline static hipError_t hipGetDeviceFlags_cu4oro(unsigned int* flags) {
    return hipCUDAErrorTohipError(cudaGetDeviceFlags(flags));
}

inline static hipError_t hipSetDeviceFlags_cu4oro(unsigned int flags) {
    return hipCUDAErrorTohipError(cudaSetDeviceFlags(flags));
}

inline static hipError_t hipEventCreateWithFlags_cu4oro(hipEvent_t* event, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaEventCreateWithFlags(event, flags));
}

inline static hipError_t hipEventQuery_cu4oro(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventQuery(event));
}

inline static hipError_t hipCtxCreate_cu4oro(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
    return hipCUResultTohipError(cuCtxCreate(ctx, flags, device));
}

inline static hipError_t hipCtxDestroy_cu4oro(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDestroy(ctx));
}

inline static hipError_t hipCtxPopCurrent_cu4oro(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxPopCurrent(ctx));
}

inline static hipError_t hipCtxPushCurrent_cu4oro(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxPushCurrent(ctx));
}

inline static hipError_t hipCtxSetCurrent_cu4oro(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxSetCurrent(ctx));
}

inline static hipError_t hipCtxGetCurrent_cu4oro(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxGetCurrent(ctx));
}

inline static hipError_t hipCtxGetDevice_cu4oro(hipDevice_t* device) {
    return hipCUResultTohipError(cuCtxGetDevice(device));
}

inline static hipError_t hipCtxGetApiVersion_cu4oro(hipCtx_t ctx, int* apiVersion) {
    return hipCUResultTohipError(cuCtxGetApiVersion(ctx, (unsigned int*)apiVersion));
}

inline static hipError_t hipCtxGetCacheConfig_cu4oro(hipFuncCache* cacheConfig) {
    return hipCUResultTohipError(cuCtxGetCacheConfig(cacheConfig));
}

inline static hipError_t hipCtxSetCacheConfig_cu4oro(hipFuncCache cacheConfig) {
    return hipCUResultTohipError(cuCtxSetCacheConfig(cacheConfig));
}

inline static hipError_t hipCtxSetSharedMemConfig_cu4oro(hipSharedMemConfig config) {
    return hipCUResultTohipError(cuCtxSetSharedMemConfig((CUsharedconfig)config));
}

inline static hipError_t hipCtxGetSharedMemConfig_cu4oro(hipSharedMemConfig* pConfig) {
    return hipCUResultTohipError(cuCtxGetSharedMemConfig((CUsharedconfig*)pConfig));
}

inline static hipError_t hipCtxSynchronize_cu4oro(void) {
    return hipCUResultTohipError(cuCtxSynchronize());
}

inline static hipError_t hipCtxGetFlags_cu4oro(unsigned int* flags) {
    return hipCUResultTohipError(cuCtxGetFlags(flags));
}

inline static hipError_t hipCtxDetach(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDetach(ctx));
}

inline static hipError_t hipDeviceGet_cu4oro(hipDevice_t* device, int ordinal) {
    return hipCUResultTohipError(cuDeviceGet(device, ordinal));
}

inline static hipError_t hipDeviceComputeCapability_cu4oro(int* major, int* minor, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceComputeCapability(major, minor, device));
}

inline static hipError_t hipDeviceGetName_cu4oro(char* name, int len, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceGetName(name, len, device));
}

inline static hipError_t hipDeviceGetUuid_cu4oro(hipUUID* uuid, hipDevice_t device) {
    if (uuid == NULL) {
      return hipErrorInvalidValue;
    }
    struct CUuuid_st CUuid;
    hipError_t err = hipCUResultTohipError(cuDeviceGetUuid(&CUuid, device));
    if (err == hipSuccess) {
      strncpy(uuid->bytes, CUuid.bytes, 16);
    }
    return err;
}

inline static hipError_t hipDeviceGetP2PAttribute_cu4oro(int* value, hipDeviceP2PAttr attr,
                                                  int srcDevice, int dstDevice) {
    return hipCUDAErrorTohipError(cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice));
}

inline static hipError_t hipDeviceGetPCIBusId_cu4oro(char* pciBusId, int len, hipDevice_t device) {
    return hipCUDAErrorTohipError(cudaDeviceGetPCIBusId(pciBusId, len, device));
}

inline static hipError_t hipDeviceGetByPCIBusId_cu4oro(int* device, const char* pciBusId) {
    return hipCUDAErrorTohipError(cudaDeviceGetByPCIBusId(device, pciBusId));
}

inline static hipError_t hipDeviceGetSharedMemConfig_cu4oro(hipSharedMemConfig* config) {
    return hipCUDAErrorTohipError(cudaDeviceGetSharedMemConfig(config));
}

inline static hipError_t hipDeviceSetSharedMemConfig_cu4oro(hipSharedMemConfig config) {
    return hipCUDAErrorTohipError(cudaDeviceSetSharedMemConfig(config));
}

inline static hipError_t hipDeviceGetLimit_cu4oro(size_t* pValue, hipLimit_t limit) {
    return hipCUDAErrorTohipError(cudaDeviceGetLimit(pValue, limit));
}

inline static hipError_t hipDeviceSetLimit_cu4oro(hipLimit_t limit, size_t value) {
    return hipCUDAErrorTohipError(cudaDeviceSetLimit(limit, value));
}

inline static hipError_t hipDeviceTotalMem_cu4oro(size_t* bytes, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceTotalMem(bytes, device));
}

inline static hipError_t hipModuleLoad_cu4oro(hipModule_t* module, const char* fname) {
    return hipCUResultTohipError(cuModuleLoad(module, fname));
}

inline static hipError_t hipModuleUnload_cu4oro(hipModule_t hmod) {
    return hipCUResultTohipError(cuModuleUnload(hmod));
}

inline static hipError_t hipModuleGetFunction_cu4oro(hipFunction_t* function, hipModule_t module,
                                              const char* kname) {
    return hipCUResultTohipError(cuModuleGetFunction(function, module, kname));
}

inline static hipError_t hipModuleGetTexRef_cu4oro(hipTexRef* pTexRef, hipModule_t hmod, const char* name){
    return hipCUResultTohipError(cuModuleGetTexRef(pTexRef, hmod, name));
}

inline static hipError_t hipFuncGetAttributes_cu4oro(hipFuncAttributes* attr, const void* func) {
    return hipCUDAErrorTohipError(cudaFuncGetAttributes(attr, func));
}

inline static hipError_t hipFuncGetAttribute_cu4oro (int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
    return hipCUResultTohipError(cuFuncGetAttribute(value, attrib, hfunc));
}

inline static hipError_t hipModuleGetGlobal_cu4oro(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                            const char* name) {
    return hipCUResultTohipError(cuModuleGetGlobal(dptr, bytes, hmod, name));
}

inline static hipError_t hipModuleLoadData_cu4oro(hipModule_t* module, const void* image) {
    return hipCUResultTohipError(cuModuleLoadData(module, image));
}

inline static hipError_t hipModuleLoadDataEx_cu4oro(hipModule_t* module, const void* image,
                                             unsigned int numOptions, hipJitOption* options,
                                             void** optionValues) {
    return hipCUResultTohipError(
        cuModuleLoadDataEx(module, image, numOptions, options, optionValues));
}

inline static hipError_t hipLaunchKernel_cu4oro(const void* function_address, dim3 numBlocks,
                                         dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                         hipStream_t stream) {
    return hipCUDAErrorTohipError(
        cudaLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream));
}

inline static hipError_t hipModuleLaunchKernel_cu4oro(hipFunction_t f, unsigned int gridDimX,
                                               unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               hipStream_t stream, void** kernelParams,
                                               void** extra) {
    return hipCUResultTohipError(cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                blockDimY, blockDimZ, sharedMemBytes, stream,
                                                kernelParams, extra));
}

inline static hipError_t hipFuncSetCacheConfig_cu4oro(const void* func, hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaFuncSetCacheConfig(func, cacheConfig));
}

#if CUDA_VERSION < CUDA_12000
__HIP_DEPRECATED inline static hipError_t hipBindTexture(size_t* offset,
                                                         struct textureReference* tex,
                                                         const void* devPtr,
                                                         const hipChannelFormatDesc* desc,
                                                         size_t size __dparm(UINT_MAX)) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

__HIP_DEPRECATED inline static hipError_t hipBindTexture2D(
    size_t* offset, struct textureReference* tex, const void* devPtr,
    const hipChannelFormatDesc* desc, size_t width, size_t height, size_t pitch) {
    return hipCUDAErrorTohipError(cudaBindTexture2D(offset, tex, devPtr, desc, width, height, pitch));
}
#endif // CUDA_VERSION < CUDA_12000


inline static hipChannelFormatDesc hipCreateChannelDesc_cu4oro(int x, int y, int z, int w,
                                                        hipChannelFormatKind f) {
    return cudaCreateChannelDesc(x, y, z, w, hipChannelFormatKindToCudaChannelFormatKind(f));
}

inline static hipChannelFormatDesc hipCreateChannelDescHalf() {
    int e = (int)sizeof(unsigned short) * 8;
    return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

inline static hipChannelFormatDesc hipCreateChannelDescHalf1() {
    int e = (int)sizeof(unsigned short) * 8;
    return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

inline static hipChannelFormatDesc hipCreateChannelDescHalf2() {
    int e = (int)sizeof(unsigned short) * 8;
    return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

inline static hipChannelFormatDesc hipCreateChannelDescHalf4() {
    int e = (int)sizeof(unsigned short) * 8;
    return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

inline static hipError_t hipCreateTextureObject_cu4oro(hipTextureObject_t* pTexObject,
                                                const hipResourceDesc* pResDesc,
                                                const hipTextureDesc* pTexDesc,
                                                const hipResourceViewDesc* pResViewDesc) {
    return hipCUDAErrorTohipError(
        cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
}

inline static hipError_t hipDestroyTextureObject_cu4oro(hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaDestroyTextureObject(textureObject));
}

inline static hipError_t hipCreateSurfaceObject_cu4oro(hipSurfaceObject_t* pSurfObject,
                                                const hipResourceDesc* pResDesc) {
    return hipCUDAErrorTohipError(cudaCreateSurfaceObject(pSurfObject, pResDesc));
}

inline static hipError_t hipDestroySurfaceObject_cu4oro(hipSurfaceObject_t surfaceObject) {
    return hipCUDAErrorTohipError(cudaDestroySurfaceObject(surfaceObject));
}

inline static hipError_t hipGetTextureObjectResourceDesc_cu4oro(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaGetTextureObjectResourceDesc( pResDesc, textureObject));
}

#if CUDA_VERSION < CUDA_12000
__HIP_DEPRECATED inline static hipError_t hipGetTextureAlignmentOffset(
    size_t* offset, const struct textureReference* texref) {
    return hipCUDAErrorTohipError(cudaGetTextureAlignmentOffset(offset,texref));
}
#endif

inline static hipError_t hipGetChannelDesc_cu4oro(hipChannelFormatDesc* desc, hipArray_const_t array)
{
    return hipCUDAErrorTohipError(cudaGetChannelDesc(desc,array));
}

inline static hipError_t hipLaunchCooperativeKernel_cu4oro(const void* f, dim3 gridDim, dim3 blockDim,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
    return hipCUDAErrorTohipError(
            cudaLaunchCooperativeKernel(f, gridDim, blockDim, kernelParams, sharedMemBytes, stream));
}

inline static hipError_t hipModuleLaunchCooperativeKernel_cu4oro(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams) {
    return hipCUResultTohipError(cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ,
                                                           blockDimX, blockDimY, blockDimZ,
                                                           sharedMemBytes, stream,kernelParams));
}

inline static hipError_t hipLaunchCooperativeKernelMultiDevice_cu4oro(hipLaunchParams* launchParamsList,
                                                 int  numDevices, unsigned int  flags) {
    return hipCUDAErrorTohipError(cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags));
}

inline static hipError_t hipModuleLaunchCooperativeKernelMultiDevice_cu4oro(
                                                       hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int  numDevices,
                                                       unsigned int  flags) {
    return hipCUResultTohipError(cuLaunchCooperativeKernelMultiDevice(launchParamsList,
                                                                      numDevices, flags));
}

inline static hipError_t hipImportExternalSemaphore_cu4oro(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  return hipCUDAErrorTohipError(cudaImportExternalSemaphore(extSem_out,(const struct cudaExternalSemaphoreHandleDesc*)semHandleDesc));
}

inline static hipError_t hipSignalExternalSemaphoresAsync_cu4oro(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream) {
  return hipCUDAErrorTohipError(cudaSignalExternalSemaphoresAsync(extSemArray, (const struct cudaExternalSemaphoreSignalParams*)paramsArray, numExtSems, stream));
}
inline static hipError_t hipWaitExternalSemaphoresAsync_cu4oro(const hipExternalSemaphore_t* extSemArray,
                                              const hipExternalSemaphoreWaitParams* paramsArray,
                                              unsigned int numExtSems, hipStream_t stream) {
  return hipCUDAErrorTohipError(cudaWaitExternalSemaphoresAsync(extSemArray, (const struct cudaExternalSemaphoreWaitParams*)paramsArray, numExtSems, stream));
}

inline static hipError_t hipDestroyExternalSemaphore_cu4oro(hipExternalSemaphore_t extSem) {
  return hipCUDAErrorTohipError(cudaDestroyExternalSemaphore(extSem));
}

inline static hipError_t hipImportExternalMemory_cu4oro(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc) {
  return hipCUDAErrorTohipError(cudaImportExternalMemory(extMem_out, (const struct cudaExternalMemoryHandleDesc*)memHandleDesc));
}

inline static hipError_t hipExternalMemoryGetMappedBuffer_cu4oro(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc) {
  return hipCUDAErrorTohipError(cudaExternalMemoryGetMappedBuffer(devPtr, extMem, (const struct cudaExternalMemoryBufferDesc*)bufferDesc));
}

inline static hipError_t hipDestroyExternalMemory_cu4oro(hipExternalMemory_t extMem) {
  return hipCUDAErrorTohipError(cudaDestroyExternalMemory(extMem));
}

inline static hipError_t hipGraphicsMapResources_cu4oro(int count, hipGraphicsResource_t* resources, hipStream_t stream  __dparm(0)) {
  return hipCUDAErrorTohipError(cudaGraphicsMapResources(count, resources, stream));
}

inline static hipError_t hipGraphicsSubResourceGetMappedArray_cu4oro(hipArray_t* array, hipGraphicsResource_t resource, unsigned int arrayIndex,
                                                              unsigned int mipLevel) {
  return hipCUDAErrorTohipError(cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel));
}

inline static hipError_t hipGraphicsResourceGetMappedPointer_cu4oro(void** devPtr, size_t* size, hipGraphicsResource_t resource) {
  return hipCUDAErrorTohipError(cudaGraphicsResourceGetMappedPointer(devPtr, size, resource));
}

inline static hipError_t hipGraphicsUnmapResources_cu4oro(int count, hipGraphicsResource_t* resources, hipStream_t stream  __dparm(0)) {
  return hipCUDAErrorTohipError(cudaGraphicsUnmapResources(count, resources, stream));
}

inline static hipError_t hipGraphicsUnregisterResource_cu4oro(hipGraphicsResource_t resource) {
  return hipCUDAErrorTohipError(cudaGraphicsUnregisterResource(resource));
}

#if CUDA_VERSION >= CUDA_11020
// ========================== HIP Stream Ordered Memory Allocator =================================
inline static hipError_t hipDeviceGetDefaultMemPool_cu4oro(hipMemPool_t* mem_pool, int device) {
  return hipCUDAErrorTohipError(cudaDeviceGetDefaultMemPool(mem_pool, device));
}

inline static hipError_t hipDeviceSetMemPool_cu4oro(int device, hipMemPool_t mem_pool) {
  return hipCUDAErrorTohipError(cudaDeviceSetMemPool(device, mem_pool));
}

inline static hipError_t hipDeviceGetMemPool_cu4oro(hipMemPool_t* mem_pool, int device) {
  return hipCUDAErrorTohipError(cudaDeviceGetMemPool(mem_pool, device));
}

inline static hipError_t hipMallocAsync_cu4oro(void** dev_ptr, size_t size, hipStream_t stream) {
  return hipCUDAErrorTohipError(cudaMallocAsync(dev_ptr, size, stream));
}

inline static hipError_t hipFreeAsync_cu4oro(void* dev_ptr, hipStream_t stream) {
  return hipCUDAErrorTohipError(cudaFreeAsync(dev_ptr, stream));
}

inline static hipError_t hipMemPoolTrimTo_cu4oro(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  return hipCUDAErrorTohipError(cudaMemPoolTrimTo(mem_pool, min_bytes_to_hold));
}

inline static hipError_t hipMemPoolSetAttribute_cu4oro(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  return hipCUDAErrorTohipError(cudaMemPoolSetAttribute(mem_pool, attr, value));
}

inline static hipError_t hipMemPoolGetAttribute_cu4oro(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  return hipCUDAErrorTohipError(cudaMemPoolGetAttribute(mem_pool, attr, value));
}

inline static hipError_t hipMemPoolSetAccess_cu4oro(
    hipMemPool_t mem_pool,
    const hipMemAccessDesc* desc_list,
    size_t count) {
  return hipCUDAErrorTohipError(cudaMemPoolSetAccess(mem_pool, desc_list, count));
}

inline static hipError_t hipMemPoolGetAccess_cu4oro(
    hipMemAccessFlags* flags,
    hipMemPool_t mem_pool,
    hipMemLocation* location) {
  return hipCUDAErrorTohipError(cudaMemPoolGetAccess(flags, mem_pool, location));
}

inline static hipError_t hipMemPoolCreate_cu4oro(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  return hipCUDAErrorTohipError(cudaMemPoolCreate(mem_pool, pool_props));
}

inline static hipError_t hipMemPoolDestroy_cu4oro(hipMemPool_t mem_pool) {
  return hipCUDAErrorTohipError(cudaMemPoolDestroy(mem_pool));
}

inline static hipError_t hipMallocFromPoolAsync_cu4oro(
    void** dev_ptr,
    size_t size,
    hipMemPool_t mem_pool,
    hipStream_t stream) {
  return hipCUDAErrorTohipError(cudaMallocFromPoolAsync(dev_ptr, size, mem_pool, stream));
}

inline static hipError_t hipMemPoolExportToShareableHandle_cu4oro(
    void*                      shared_handle,
    hipMemPool_t               mem_pool,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags) {
  return hipCUDAErrorTohipError(cudaMemPoolExportToShareableHandle(
            shared_handle, mem_pool, handle_type, flags));
}

inline static hipError_t hipMemPoolImportFromShareableHandle_cu4oro(
    hipMemPool_t*              mem_pool,
    void*                      shared_handle,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags) {
  return hipCUDAErrorTohipError(cudaMemPoolImportFromShareableHandle(
            mem_pool, shared_handle, handle_type, flags));
}

inline static hipError_t hipMemPoolExportPointer_cu4oro(hipMemPoolPtrExportData* export_data, void* ptr) {
  return hipCUDAErrorTohipError(cudaMemPoolExportPointer(export_data, ptr));
}

inline static hipError_t hipMemPoolImportPointer_cu4oro(
    void**                   ptr,
    hipMemPool_t             mem_pool,
    hipMemPoolPtrExportData* export_data) {
  return hipCUDAErrorTohipError(cudaMemPoolImportPointer(ptr, mem_pool, export_data));
}
#endif // CUDA_VERSION >= CUDA_11020

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

template<class T>
inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor_cu4oro(int* numBlocks,
                                                                      T func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                            blockSize, dynamicSMemSize));
}

template <class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                                           dynamicSMemSize, blockSizeLimit));
}

template <typename UnaryFunction, class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* min_grid_size,
                                                                                int* block_size,
                                                                                T func,
                                                                                UnaryFunction block_size_to_dynamic_smem_size,
                                                                                int block_size_limit = 0,
                                                                                unsigned int flags = 0) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(min_grid_size, block_size, func,
                                                    block_size_to_dynamic_smem_size, block_size_limit,flags));
}

template <class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0, unsigned int  flags = 0) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                                           dynamicSMemSize, blockSizeLimit, flags));
}

template <class T>
inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cu4oro( int* numBlocks, T func,
                                              int  blockSize, size_t dynamicSMemSize,unsigned int flags) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func,
                                                                 blockSize, dynamicSMemSize, flags));
}

#if CUDA_VERSION < CUDA_12000
template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, const hipChannelFormatDesc& desc,
                                        size_t size = UINT_MAX) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipUnbindTexture(struct texture<T, dim, readMode>* tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipUnbindTexture(struct texture<T, dim, readMode>& tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>& tex, hipArray_const_t array,
    const hipChannelFormatDesc& desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>* tex, hipArray_const_t array,
    const hipChannelFormatDesc* desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>& tex, hipArray_const_t array) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array));
}
#endif   // CUDA_VERSION < CUDA_12000

template <class T>
inline static hipChannelFormatDesc hipCreateChannelDesc_cu4oro() {
    return cudaCreateChannelDesc<T>();
}

template <class T>
inline static hipError_t hipLaunchCooperativeKernel_cu4oro(T f, dim3 gridDim, dim3 blockDim,
                                             void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
    return hipCUDAErrorTohipError(
            cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(f), gridDim, blockDim, kernelParams, sharedMemBytes, stream));
}

inline static hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject,
                                            const HIP_RESOURCE_DESC* pResDesc,
                                            const HIP_TEXTURE_DESC* pTexDesc,
                                            const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
    return hipCUResultTohipError(cuTexObjectCreate((CUtexObject*)pTexObject, pResDesc, pTexDesc, pResViewDesc));
}

inline static hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
    return hipCUResultTohipError(cuTexObjectDestroy((CUtexObject)texObject));
}

inline static hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject) {
    return hipCUResultTohipError(cuTexObjectGetResourceDesc(pResDesc, (CUtexObject)texObject));
}

inline static hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc, hipTextureObject_t texObject) {
    return hipCUResultTohipError(cuTexObjectGetResourceViewDesc(pResViewDesc, (CUtexObject)texObject));
}

inline static hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject) {
    return hipCUResultTohipError(cuTexObjectGetTextureDesc(pTexDesc, (CUtexObject)texObject));
}

__HIP_DEPRECATED inline static hipError_t hipTexRefSetAddressMode(hipTexRef hTexRef, int dim, hipAddress_mode am){
    return hipCUResultTohipError(cuTexRefSetAddressMode(hTexRef,dim,am));
}

__HIP_DEPRECATED inline static hipError_t hipTexRefSetFilterMode(hipTexRef hTexRef, hipFilter_mode fm){
    return hipCUResultTohipError(cuTexRefSetFilterMode(hTexRef,fm));
}

inline static hipError_t hipTexRefSetAddress(size_t *ByteOffset, hipTexRef hTexRef, hipDeviceptr_t dptr, size_t bytes){
    return hipCUResultTohipError(cuTexRefSetAddress(ByteOffset,hTexRef,dptr,bytes));
}

inline static hipError_t hipTexRefSetAddress2D(hipTexRef hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, hipDeviceptr_t dptr, size_t Pitch){
    return hipCUResultTohipError(cuTexRefSetAddress2D(hTexRef,desc,dptr,Pitch));
}

__HIP_DEPRECATED inline static hipError_t hipTexRefSetFormat(hipTexRef hTexRef, hipArray_Format fmt, int NumPackedComponents){
    return hipCUResultTohipError(cuTexRefSetFormat(hTexRef,fmt,NumPackedComponents));
}

__HIP_DEPRECATED inline static hipError_t hipTexRefSetFlags(hipTexRef hTexRef, unsigned int Flags){
    return hipCUResultTohipError(cuTexRefSetFlags(hTexRef,Flags));
}

__HIP_DEPRECATED inline static hipError_t hipTexRefSetArray(hipTexRef hTexRef, hiparray hArray, unsigned int Flags){
    return hipCUResultTohipError(cuTexRefSetArray(hTexRef,hArray,Flags));
}

inline static hipError_t hipArrayCreate(hiparray* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray){
    return hipCUResultTohipError(cuArrayCreate(pHandle, pAllocateArray));
}

inline static hipError_t hipArrayDestroy(hiparray hArray){
    return hipCUResultTohipError(cuArrayDestroy(hArray));
}

inline static hipError_t hipArray3DCreate(hiparray* pHandle,
                                          const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray){
    return hipCUResultTohipError(cuArray3DCreate(pHandle, pAllocateArray));
}

inline static hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent,
                                          unsigned int* flags, hipArray* array) {
    return hipCUDAErrorTohipError(cudaArrayGetInfo(desc, extent, flags, array));
}

inline static hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                               hipArray* array) {
    return hipCUResultTohipError(cuArrayGetDescriptor(pArrayDescriptor, (CUarray)array));
}

inline static hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor,
                                                 hipArray* array) {
    return hipCUResultTohipError(cuArray3DGetDescriptor(pArrayDescriptor, (CUarray)array));
}

inline static hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
    return hipCUDAErrorTohipError(cudaStreamBeginCapture(stream, mode));
}

inline static hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
    return hipCUDAErrorTohipError(cudaStreamEndCapture(stream, pGraph));
}

inline static hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaGraphCreate(pGraph, flags));
}

inline static hipError_t hipGraphDestroy(hipGraph_t graph) {
    return hipCUDAErrorTohipError(cudaGraphDestroy(graph));
}

inline static hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec) {
    return hipCUDAErrorTohipError(cudaGraphExecDestroy(pGraphExec));
}

inline static hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                             hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                             size_t bufferSize) {
    return hipCUDAErrorTohipError(
        cudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize));
}

#if CUDA_VERSION >= CUDA_11040
inline static hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                                      unsigned long long flags) {
    return hipCUDAErrorTohipError(cudaGraphInstantiateWithFlags(pGraphExec, graph, flags));
}

inline hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                          const hipGraphNode_t* pDependencies,
                                          size_t numDependencies,
                                          hipMemAllocNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphAddMemAllocNode(
        pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
}

inline hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node,
                                                hipMemAllocNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphMemAllocNodeGetParams(node, pNodeParams));
}

inline hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, void* dev_ptr) {
    return hipCUDAErrorTohipError(cudaGraphAddMemFreeNode(
        pGraphNode, graph, pDependencies, numDependencies, dev_ptr));
}

inline hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
    return hipCUDAErrorTohipError(cudaGraphMemFreeNodeGetParams(node, dev_ptr));
}
#endif
inline static hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaGraphLaunch(graphExec, stream));
}

inline static hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                               const hipGraphNode_t* pDependencies,
                                               size_t numDependencies,
                                               const hipKernelNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(
        cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
}

inline static hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                               const hipGraphNode_t* pDependencies,
                                               size_t numDependencies,
                                               const hipMemcpy3DParms* pCopyParams) {
    return hipCUDAErrorTohipError(
        cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind));
}
#endif

inline static hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                               const hipGraphNode_t* pDependencies,
                                               size_t numDependencies,
                                               const hipMemsetParams* pMemsetParams) {
    return hipCUDAErrorTohipError(
        cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams));
}

inline static hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes,
                                          size_t* numNodes) {
    return hipCUDAErrorTohipError(cudaGraphGetNodes(graph, nodes, numNodes));
}

inline static hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                              size_t* pNumRootNodes) {
    return hipCUDAErrorTohipError(cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes));
}

inline static hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node,
                                                     hipKernelNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphKernelNodeGetParams(node, pNodeParams));
}

inline static hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                                     const hipKernelNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphKernelNodeSetParams(node, pNodeParams));
}

inline static hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                                        const hipKernelNodeAttrValue* value) {
    return hipCUDAErrorTohipError(cudaGraphKernelNodeSetAttribute(hNode, attr, value));
}

inline static hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                                        hipKernelNodeAttrValue* value) {
    return hipCUDAErrorTohipError(cudaGraphKernelNodeGetAttribute(hNode, attr, value));
}

inline static hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node,
                                                     hipMemcpy3DParms* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphMemcpyNodeGetParams(node, pNodeParams));
}

inline static hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node,
                                                     const hipMemcpy3DParms* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphMemcpyNodeSetParams(node, pNodeParams));
}

inline static hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,
                                                     hipMemsetParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphMemsetNodeGetParams(node, pNodeParams));
}

inline static hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,
                                                     const hipMemsetParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphMemsetNodeSetParams(node, pNodeParams));
}

inline static hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
    return hipCUDAErrorTohipError(cudaThreadExchangeStreamCaptureMode(mode));
}

inline static hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec,
                                                         hipGraphNode_t node,
                                                         const hipKernelNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams));
}

inline static hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                                 const hipGraphNode_t* to, size_t numDependencies) {
    return hipCUDAErrorTohipError(cudaGraphAddDependencies(graph, from, to, numDependencies));
}

inline static hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                              const hipGraphNode_t* pDependencies,
                                              size_t numDependencies) {
    return hipCUDAErrorTohipError(
      cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies));
}

inline static hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value,
                                               unsigned int flags) {
    if (value < 0) {
        printf("Warning! value is negative, CUDA accept positive values\n");
    }
    return hipCUResultTohipError(cuStreamWriteValue32(stream, reinterpret_cast<CUdeviceptr>(ptr),
                                                      static_cast<cuuint32_t>(value), flags));
}

inline static hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value,
                                               unsigned int flags) {
    if (value < 0) {
        printf("Warning! value is negative, CUDA accept positive values\n");
    }
    return hipCUResultTohipError(cuStreamWriteValue64(stream, reinterpret_cast<CUdeviceptr>(ptr),
                                                      static_cast<cuuint64_t>(value), flags));
}

inline static hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value,
                                              unsigned int flags,
                                              uint32_t mask __dparm(0xFFFFFFFF)) {
    if (value < 0) {
        printf("Warning! value is negative, CUDA accept positive values\n");
    }
    if (mask != STREAM_OPS_WAIT_MASK_32) {
        printf("Warning! mask will not have impact as CUDA ignores it.\n");
    }
    return hipCUResultTohipError(cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(ptr),
                                                     static_cast<cuuint32_t>(value), flags));
}

inline static hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value,
                                              unsigned int flags,
                                              uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF)) {
    if (value < 0) {
        printf("Warning! value is negative, CUDA accept positive values\n");
    }
    if (mask != STREAM_OPS_WAIT_MASK_64) {
        printf("Warning! mask will not have impact as CUDA ignores it.\n");
    }
    return hipCUResultTohipError(cuStreamWaitValue64(stream, reinterpret_cast<CUdeviceptr>(ptr),
                                                     static_cast<cuuint64_t>(value), flags));
}

inline static hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                                    const hipGraphNode_t* to,
                                                    size_t numDependencies) {
    return hipCUDAErrorTohipError(cudaGraphRemoveDependencies(graph, from, to, numDependencies));
}


inline static hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from,
                                          hipGraphNode_t* to, size_t* numEdges) {
    return hipCUDAErrorTohipError(cudaGraphGetEdges(graph, from, to, numEdges));
}

inline static hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,
                                                     hipGraphNode_t* pDependencies,
                                                     size_t* pNumDependencies) {
    return hipCUDAErrorTohipError(
        cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies));
}

inline static hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,
                                                       hipGraphNode_t* pDependentNodes,
                                                       size_t* pNumDependentNodes) {
    return hipCUDAErrorTohipError(
        cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes));
}

inline static hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
    return hipCUDAErrorTohipError(cudaGraphNodeGetType(node, pType));
}

inline static hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
    return hipCUDAErrorTohipError(cudaGraphDestroyNode(node));
}

inline static hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
    return hipCUDAErrorTohipError(cudaGraphClone(pGraphClone, originalGraph));
}

inline static hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                                 hipGraph_t clonedGraph) {
    return hipCUDAErrorTohipError(cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph));
}

inline static hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                   const hipGraphNode_t* pDependencies,
                                                   size_t numDependencies, hipGraph_t childGraph) {
    return hipCUDAErrorTohipError(
        cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph));
}

inline static hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
    return hipCUDAErrorTohipError(cudaGraphChildGraphNodeGetGraph(node, pGraph));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,
                                                             hipGraphNode_t node,
                                                             hipGraph_t childGraph) {
    return hipCUDAErrorTohipError(
        cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph));
}
#endif

inline static hipError_t hipStreamGetCaptureInfo(hipStream_t stream,
                                                 hipStreamCaptureStatus* pCaptureStatus,
                                                 unsigned long long* pId) {
    return hipCUDAErrorTohipError(cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId));
}

#if CUDA_VERSION >= CUDA_11030
inline static hipError_t hipStreamGetCaptureInfo_v2(
    hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out __dparm(0), hipGraph_t* graph_out __dparm(0),
    const hipGraphNode_t** dependencies_out __dparm(0), size_t* numDependencies_out __dparm(0)) {
    return hipCUResultTohipError(cuStreamGetCaptureInfo_v2(
        stream, reinterpret_cast<CUstreamCaptureStatus *>(captureStatus_out),
        reinterpret_cast<cuuint64_t *>(id_out), graph_out,
        dependencies_out, numDependencies_out));
}
#endif

inline static hipError_t hipStreamIsCapturing(hipStream_t stream,
                                              hipStreamCaptureStatus* pCaptureStatus) {
    return hipCUDAErrorTohipError(cudaStreamIsCapturing(stream, pCaptureStatus));
}

#if CUDA_VERSION >= CUDA_11030
inline static hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,
                                                            hipGraphNode_t* dependencies,
                                                            size_t numDependencies,
                                                            unsigned int flags __dparm(0)) {
    return hipCUDAErrorTohipError(cudaStreamUpdateCaptureDependencies(stream, dependencies,
                                                                      numDependencies, flags));
}
#endif

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                    const hipGraphNode_t* pDependencies,
                                                    size_t numDependencies, hipEvent_t event) {
    return hipCUDAErrorTohipError(
        cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event));
}

inline static hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                  const hipGraphNode_t* pDependencies,
                                                  size_t numDependencies, hipEvent_t event) {
    return hipCUDAErrorTohipError(
        cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event));
}
#endif

inline static hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                             const hipGraphNode_t* pDependencies,
                                             size_t numDependencies,
                                             const hipHostNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(
        cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode,
                                                         hipGraph_t graph,
                                                         const hipGraphNode_t* pDependencies,
                                                         size_t numDependencies, void* dst,
                                                         const void* symbol, size_t count,
                                                         size_t offset, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaGraphAddMemcpyNodeFromSymbol(
        pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind));
}

inline static hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                                       const hipGraphNode_t* pDependencies,
                                                       size_t numDependencies, const void* symbol,
                                                       const void* src, size_t count, size_t offset,
                                                       hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaGraphAddMemcpyNodeToSymbol(
        pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind));
}

inline static hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaGraphEventRecordNodeSetEvent(node, event));
}

inline static hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
    return hipCUDAErrorTohipError(cudaGraphEventWaitNodeGetEvent(node, event_out));
}

inline static hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaGraphEventWaitNodeSetEvent(node, event));
}
#endif

inline static hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec,
                                                       hipGraphNode_t node,
                                                       const hipHostNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams));
}

inline static hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec,
                                                         hipGraphNode_t node,
                                                         hipMemcpy3DParms* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec,
                                                           hipGraphNode_t node, void* dst,
                                                           const void* src, size_t count,
                                                           hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind));
}

inline static hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec,
                                                                   hipGraphNode_t node, void* dst,
                                                                   const void* symbol, size_t count,
                                                                   size_t offset,
                                                                   hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaGraphExecMemcpyNodeSetParamsFromSymbol(
        hGraphExec, node, dst, symbol, count, offset, kind));
}

inline static hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(
    hipGraphExec_t hGraphExec, hipGraphNode_t node, const void* symbol, const void* src,
    size_t count, size_t offset, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaGraphExecMemcpyNodeSetParamsToSymbol(
        hGraphExec, node, symbol, src, count, offset, kind));
}
#endif

inline static hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,
                                                         hipGraphNode_t node,
                                                         const hipMemsetParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams));
}

inline static hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                                            hipGraphNode_t* hErrorNode_out,
                                            hipGraphExecUpdateResult* updateResult_out) {
    return hipCUDAErrorTohipError(
        cudaGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst,
                                                               const void* symbol, size_t count,
                                                               size_t offset, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind));
}

inline static hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node,
                                                             const void* symbol, const void* src,
                                                             size_t count, size_t offset,
                                                             hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind));
}

inline static hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,
                                                         hipEvent_t* event_out) {
    return hipCUDAErrorTohipError(cudaGraphEventRecordNodeGetEvent(node, event_out));
}
#endif

inline static hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,
                                                   hipHostNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphHostNodeGetParams(node, pNodeParams));
}

#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst,
                                                       const void* src, size_t count,
                                                       hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind));
}

inline static hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,
                                                             hipGraphNode_t hNode,
                                                             hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event));
}

inline static hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,
                                                           hipGraphNode_t hNode, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event));
}

inline static hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
    return hipCUDAErrorTohipError(cudaDeviceGetGraphMemAttribute(device, attr, value));
}

inline static hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
    return hipCUDAErrorTohipError(cudaDeviceSetGraphMemAttribute(device, attr, value));
}

inline static hipError_t hipDeviceGraphMemTrim(int device) {
    return hipCUDAErrorTohipError(cudaDeviceGraphMemTrim(device));
}

inline static hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
    return hipCUDAErrorTohipError(cudaLaunchHostFunc(stream, fn, userData));
}

inline static hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                                             unsigned int initialRefcount, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags));
}


inline static hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count __dparm(1)) {
    return hipCUDAErrorTohipError(cudaUserObjectRelease(object, count));
}


inline static hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count __dparm(1)) {
    return hipCUDAErrorTohipError(cudaUserObjectRelease(object, count));
}

inline static hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count __dparm(1), unsigned int flags __dparm(0)) {
    return hipCUDAErrorTohipError(cudaGraphRetainUserObject(graph, object, count, flags));
}

inline static hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count __dparm(1)) {
    return hipCUDAErrorTohipError(cudaGraphReleaseUserObject(graph, object, count));
}
#endif

inline static hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node,
                                                   const hipHostNodeParams* pNodeParams) {
    return hipCUDAErrorTohipError(cudaGraphHostNodeSetParams(node, pNodeParams));
}
#if CUDA_VERSION >= CUDA_11030
inline static hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path,
                                               unsigned int flags) {
    return hipCUDAErrorTohipError(cudaGraphDebugDotPrint(graph, path, flags));
}
#endif
#if CUDA_VERSION >= CUDA_11000
inline static hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc,
                                                          hipGraphNode_t hDst) {
    return hipCUDAErrorTohipError(cudaGraphKernelNodeCopyAttributes(hSrc, hDst));
}
#endif
#if CUDA_VERSION >= CUDA_11060
inline static hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                                unsigned int isEnabled) {
    return hipCUDAErrorTohipError(cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled));
}

inline static hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                                unsigned int* isEnabled) {
    return hipCUDAErrorTohipError(cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled));
}
#endif
#if CUDA_VERSION >= CUDA_11010
inline static hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaGraphUpload(graphExec, stream));
}
#endif
#endif  //__CUDACC__

#endif  // HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H
