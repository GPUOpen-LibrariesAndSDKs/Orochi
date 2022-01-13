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
typedef unsigned long long ppTextureObject;


ppError PPAPI ppGetErrorName(ppError error, const char** pStr);
ppError PPAPI ppInit(unsigned int Flags);
ppError PPAPI ppDriverGetVersion(int* driverVersion);
ppError PPAPI ppGetDevice(int* device);
ppError PPAPI ppGetDeviceCount(int* count);
//ppError PPAPI ppGetDeviceProperties(ppDeviceProp_t* props, int deviceId);
ppError PPAPI ppDeviceGet(ppDevice* device, int ordinal);
ppError PPAPI ppDeviceGetName(char* name, int len, ppDevice dev);
//ppError PPAPI ppDeviceGetAttribute(int* pi, ppDeviceAttribute attrib, ppDevice dev);
ppError PPAPI ppDeviceComputeCapability(int* major, int* minor, ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxRetain(ppCtx* pctx, ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxRelease(ppDevice dev);
ppError PPAPI ppDevicePrimaryCtxSetFlags(ppDevice dev, unsigned int flags);
ppError PPAPI ppDevicePrimaryCtxGetState(ppDevice dev, unsigned int* flags, int* active);
ppError PPAPI ppDevicePrimaryCtxReset(ppDevice dev);
ppError PPAPI ppCtxCreate(ppCtx* pctx, unsigned int flags, ppDevice dev);
ppError PPAPI ppCtxDestroy(ppCtx ctx);


enum {
	PP_SUCCESS = 0,
	PP_ERROR_OPEN_FAILED = -1,
	PP_ERROR_ATEXIT_FAILED = -2,
	PP_ERROR_OLD_DRIVER = -3,
};


int ppInitialize( Api api, ppU32 flags );

