#include <Pop/Pop.h>
#include <contrib/cuew/include/cuew.h>
#include <contrib/hipew/include/hipew.h>
#include <stdio.h>
#include <string.h>


static Api s_api = API_HIP;

int ppInitialize( Api api, ppU32 flags )
{
	s_api = api;
	if( api == API_CUDA )
		return cuewInit( CUEW_INIT_CUDA | CUEW_INIT_NVRTC );
	if( api == API_HIP )
		return hipewInit( HIPEW_INIT_HIP );
	return PP_ERROR_OPEN_FAILED;
}


//=================================

inline
ppError hip2pp( hipError_t a )
{
	return (ppError)a;
}
inline
ppError cu2pp( CUresult a )
{
	return (ppError)a;
}
inline
CUcontext* ppCtx2cu( ppCtx* a )
{
	return (CUcontext*)a;
}
inline
hipCtx_t* ppCtx2hip( ppCtx* a )
{
	return (hipCtx_t*)a;
}
inline
pprtcResult hiprtc2pp( hiprtcResult a )
{
	return (pprtcResult)a;
}
inline
pprtcResult nvrtc2pp( nvrtcResult a )
{
	return (pprtcResult)a;
}

#define __PP_FUNC1( cuname, hipname ) if( s_api == API_CUDA ) return cu2pp( cu##cuname ); if( s_api == API_HIP ) return hip2pp( hip##hipname );
#define __PP_FUNC( name ) if( s_api == API_CUDA ) return cu2pp( cu##name ); if( s_api == API_HIP ) return hip2pp( hip##name );
#define __PP_CTXT_FUNC( name ) __PP_FUNC1(Ctx##name, name)
//#define __PP_CTXT_FUNC( name ) if( s_api == API_CUDA ) return cu2pp( cuCtx##name ); if( s_api == API_HIP ) return hip2pp( hip##name );
#define __PPRTC_FUNC1( cuname, hipname ) if( s_api == API_CUDA ) return nvrtc2pp( nvrtc##cuname ); if( s_api == API_HIP ) return hiprtc2pp( hiprtc##hipname );


ppError PPAPI ppGetErrorName(ppError error, const char** pStr)
{
	__PP_FUNC1(GetErrorName((CUresult)error, pStr),
		GetErrorName((hipError_t)error, pStr));
	return ppErrorUnknown;
}
ppError PPAPI ppGetErrorString(ppError error, const char** pStr)
{
	__PP_FUNC1(GetErrorString((CUresult)error, pStr),
		GetErrorString((hipError_t)error, pStr));
	return ppErrorUnknown;
}

ppError PPAPI ppInit(unsigned int Flags)
{
	__PP_FUNC( Init(Flags) );
/*
	if( s_api == API_CUDA )
		return cu2pp( cuInit( Flags ) );
	if( s_api == API_HIP )
		return hip2pp( hipInit( Flags ) );
*/
	return ppErrorUnknown;
}
ppError PPAPI ppDriverGetVersion(int* driverVersion)
{
	__PP_FUNC( DriverGetVersion(driverVersion) );
	return ppErrorUnknown;
}
ppError PPAPI ppGetDevice(int* device)
{
	__PP_CTXT_FUNC( GetDevice(device) );
	return ppErrorUnknown;
}
ppError PPAPI ppGetDeviceCount(int* count)
{
	__PP_FUNC1( DeviceGetCount(count), GetDeviceCount(count) );
	return ppErrorUnknown;
}
ppError PPAPI ppGetDeviceProperties(ppDeviceProp* props, int deviceId)
{
	if( s_api == API_CUDA )
	{
		CUdevprop p;
		cuDeviceGetProperties( &p, deviceId );
		char name[128];
		cuDeviceGetName( name, 128, deviceId );
		strcpy( props->name, name );
		strcpy( props->gcnArchName, "" );
		printf("todo. implement me\n");
		return ppSuccess;
	}
	return hip2pp( hipGetDeviceProperties( (hipDeviceProp_t*)props, deviceId ) );
}
ppError PPAPI ppDeviceGet(ppDevice* device, int ordinal)
{
	__PP_FUNC( DeviceGet(device, ordinal) );
	return ppErrorUnknown;
}
ppError PPAPI ppDeviceGetName(char* name, int len, ppDevice dev)
{
	__PP_FUNC( DeviceGetName(name, len, dev) );
	return ppErrorUnknown;
}

ppError PPAPI ppDeviceGetAttribute(int* pi, ppDeviceAttribute attrib, ppDevice dev)
{
	__PP_FUNC1(DeviceGetAttribute(pi, (CUdevice_attribute)attrib, dev),
		DeviceGetAttribute(pi, (hipDeviceAttribute_t)attrib, dev));
	return ppErrorUnknown;
}

ppError PPAPI ppDeviceComputeCapability(int* major, int* minor, ppDevice dev)
{
	return ppErrorUnknown;
}
ppError PPAPI ppDevicePrimaryCtxRetain(ppCtx* pctx, ppDevice dev)
{
	return ppErrorUnknown;
}
ppError PPAPI ppDevicePrimaryCtxRelease(ppDevice dev)
{
	return ppErrorUnknown;
}
ppError PPAPI ppDevicePrimaryCtxSetFlags(ppDevice dev, unsigned int flags)
{
	return ppErrorUnknown;
}
ppError PPAPI ppDevicePrimaryCtxGetState(ppDevice dev, unsigned int* flags, int* active)
{
	return ppErrorUnknown;
}
ppError PPAPI ppDevicePrimaryCtxReset(ppDevice dev)
{
	return ppErrorUnknown;
}
ppError PPAPI ppCtxCreate(ppCtx* pctx, unsigned int flags, ppDevice dev)
{
	__PP_FUNC1( CtxCreate(ppCtx2cu(pctx),flags,dev), CtxCreate(ppCtx2hip(pctx),flags,dev) );
	return ppErrorUnknown;
}
ppError PPAPI ppCtxDestroy(ppCtx ctx)
{
	return ppErrorUnknown;
}
/*
ppError PPAPI ppCtxPushCurrent(ppCtx ctx);
ppError PPAPI ppCtxPopCurrent(ppCtx* pctx);
ppError PPAPI ppCtxSetCurrent(ppCtx ctx);
ppError PPAPI ppCtxGetCurrent(ppCtx* pctx);
ppError PPAPI ppCtxGetDevice(ppDevice* device);
ppError PPAPI ppCtxGetFlags(unsigned int* flags);
*/
ppError PPAPI ppCtxSynchronize(void)
{
	__PP_FUNC( CtxSynchronize() );
	return ppErrorUnknown;
}
ppError PPAPI ppDeviceSynchronize(void)
{
	__PP_FUNC1( CtxSynchronize(), DeviceSynchronize() );
	return ppErrorUnknown;
}
//ppError PPAPI ppCtxGetCacheConfig(hipFuncCache_t* pconfig);
//ppError PPAPI ppCtxSetCacheConfig(hipFuncCache_t config);
//ppError PPAPI ppCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
//ppError PPAPI ppCtxSetSharedMemConfig(hipSharedMemConfig config);
ppError PPAPI ppCtxGetApiVersion(ppCtx ctx, unsigned int* version)
{
	__PP_FUNC1( CtxGetApiVersion(*ppCtx2cu(&ctx), version ), CtxGetApiVersion(*ppCtx2hip(&ctx), version ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleLoad(ppModule* module, const char* fname)
{
	__PP_FUNC1( ModuleLoad( (CUmodule*)module, fname ), ModuleLoad( (hipModule_t*)module, fname ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleLoadData(ppModule* module, const void* image)
{
	__PP_FUNC1( ModuleLoadData( (CUmodule*)module, image ), ModuleLoadData( (hipModule_t*)module, image ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleLoadDataEx(ppModule* module, const void* image, unsigned int numOptions, hipJitOption* options, void** optionValues)
{
	__PP_FUNC1( ModuleLoadDataEx( (CUmodule*)module, image, numOptions, (CUjit_option*)options, optionValues ),
		ModuleLoadDataEx( (hipModule_t*)module, image, numOptions, (hipJitOption*)options, optionValues ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleUnload(ppModule module)
{
	__PP_FUNC1( ModuleUnload( (CUmodule)module ), ModuleUnload( (hipModule_t)module ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleGetFunction(ppFunction* hfunc, ppModule hmod, const char* name)
{
	__PP_FUNC1( ModuleGetFunction( (CUfunction*)hfunc, (CUmodule)hmod, name ), 
		ModuleGetFunction( (hipFunction_t*)hfunc, (hipModule_t)hmod, name ) );
	return ppErrorUnknown;
}
ppError PPAPI ppModuleGetGlobal(ppDeviceptr* dptr, size_t* bytes, ppModule hmod, const char* name)
{
	__PP_FUNC1( ModuleGetGlobal( dptr, bytes, (CUmodule)hmod, name ), 
		ModuleGetGlobal( dptr, bytes, (hipModule_t)hmod, name ) );
	return ppErrorUnknown;
}
//ppError PPAPI ppModuleGetTexRef(textureReference** pTexRef, ppModule hmod, const char* name);
ppError PPAPI ppMemGetInfo(size_t* free, size_t* total)
{
	return ppErrorUnknown;
}
ppError PPAPI ppMalloc(ppDeviceptr* dptr, size_t bytesize)
{
	__PP_FUNC1( MemAlloc(dptr, bytesize), Malloc( dptr, bytesize ) );
	return ppErrorUnknown;
}
ppError PPAPI ppMemAllocPitch(ppDeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
	return ppErrorUnknown;
}
ppError PPAPI ppFree(ppDeviceptr dptr)
{
	__PP_FUNC1( MemFree( dptr ), Free( dptr ) );
	return ppErrorUnknown;
}

//-------------------
ppError PPAPI ppMemcpyHtoD(ppDeviceptr dstDevice, void* srcHost, size_t ByteCount)
{
	__PP_FUNC1( MemcpyHtoD( dstDevice, srcHost, ByteCount ),
		MemcpyHtoD( dstDevice, srcHost, ByteCount ) );
	return ppErrorUnknown;
}
ppError PPAPI ppMemcpyDtoH(void* dstHost, ppDeviceptr srcDevice, size_t ByteCount)
{
	__PP_FUNC1( MemcpyDtoH( dstHost, srcDevice, ByteCount ),
		MemcpyDtoH( dstHost, srcDevice, ByteCount ) );
	return ppErrorUnknown;
}
ppError PPAPI ppMemcpyDtoD(ppDeviceptr dstDevice, ppDeviceptr srcDevice, size_t ByteCount)
{
	__PP_FUNC( MemcpyDtoD( dstDevice, srcDevice, ByteCount ) );
	return ppErrorUnknown;
}

ppError PPAPI ppMemset(ppDeviceptr dstDevice, unsigned int ui, size_t N)
{
	__PP_FUNC( MemsetD32( dstDevice, ui, N ) );
	return ppErrorUnknown;
}

ppError PPAPI ppMemsetD8(ppDeviceptr dstDevice, unsigned char ui, size_t N)
{
	__PP_FUNC(MemsetD8(dstDevice, ui, N));
	return ppErrorUnknown;
}
ppError PPAPI ppMemsetD16(ppDeviceptr dstDevice, unsigned short ui, size_t N)
{
	__PP_FUNC(MemsetD16(dstDevice, ui, N));
	return ppErrorUnknown;
}
ppError PPAPI ppMemsetD32(ppDeviceptr dstDevice, unsigned int ui, size_t N)
{
	__PP_FUNC(MemsetD32(dstDevice, ui, N));
	return ppErrorUnknown;
}

//-------------------
ppError PPAPI ppModuleLaunchKernel(ppFunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, ppStream hStream, void** kernelParams, void** extra)
{
	__PP_FUNC1( LaunchKernel( (CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra ),
		ModuleLaunchKernel( (hipFunction_t)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (hipStream_t)hStream, kernelParams, extra ) );
	return ppErrorUnknown;
}
//-------------------
pprtcResult PPAPI pprtcGetErrorString(pprtcResult result)
{
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcAddNameExpression(pprtcProgram prog, const char* name_expression)
{
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcCompileProgram(pprtcProgram prog, int numOptions, const char** options)
{
	__PPRTC_FUNC1( CompileProgram( (nvrtcProgram)prog, numOptions, options ),
		CompileProgram( (hiprtcProgram)prog, numOptions, options ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcCreateProgram(pprtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames)
{
	__PPRTC_FUNC1( CreateProgram( (nvrtcProgram*)prog, src, name, numHeaders, headers, includeNames ), 
		CreateProgram( (hiprtcProgram*)prog, src, name, numHeaders, headers, includeNames ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcDestroyProgram(pprtcProgram* prog)
{
	__PPRTC_FUNC1( DestroyProgram( (nvrtcProgram*)prog), 
		DestroyProgram( (hiprtcProgram*)prog ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcGetLoweredName(pprtcProgram prog, const char* name_expression, const char** lowered_name)
{
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcGetProgramLog(pprtcProgram prog, char* log)
{
	__PPRTC_FUNC1( GetProgramLog( (nvrtcProgram)prog, log ), 
		GetProgramLog( (hiprtcProgram)prog, log ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcGetProgramLogSize(pprtcProgram prog, size_t* logSizeRet)
{
	__PPRTC_FUNC1( GetProgramLogSize( (nvrtcProgram)prog, logSizeRet), 
		GetProgramLogSize( (hiprtcProgram)prog, logSizeRet ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcGetCode(pprtcProgram prog, char* code)
{
	__PPRTC_FUNC1( GetPTX( (nvrtcProgram)prog, code ), 
		GetCode( (hiprtcProgram)prog, code ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}
pprtcResult PPAPI pprtcGetCodeSize(pprtcProgram prog, size_t* codeSizeRet)
{
	__PPRTC_FUNC1( GetPTXSize( (nvrtcProgram)prog, codeSizeRet ), 
		GetCodeSize( (hiprtcProgram)prog, codeSizeRet ) );
	return PPRTC_ERROR_INTERNAL_ERROR;
}

//-------------------

// Implementation of ppPointerGetAttributes is hacky due to differences between CUDA and HIP
ppError PPAPI ppPointerGetAttributes(ppPointerAttribute* attr, ppDeviceptr dptr)
{
	if (s_api == API_CUDA)
	{
		unsigned int data;
		return cu2pp(cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr));
	}
	if (s_api == API_HIP) 
		return hip2pp(hipPointerGetAttributes((hipPointerAttribute_t*)attr, (void*)dptr));

	return ppErrorUnknown;
}

//-----------------
ppError PPAPI ppStreamCreate(ppStream* stream)
{
	__PP_FUNC1( StreamCreate((CUstream*)stream, CU_STREAM_DEFAULT),
		StreamCreate((hipStream_t*)stream) );
	return ppErrorUnknown;
}

