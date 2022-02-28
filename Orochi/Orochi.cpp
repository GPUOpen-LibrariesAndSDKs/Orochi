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

#include <Orochi/Orochi.h>
#include <contrib/cuew/include/cuew.h>
#include <contrib/hipew/include/hipew.h>
#include <stdio.h>
#include <string.h>


thread_local static oroApi s_api = ORO_API_HIP;
static oroU32 s_loadedApis = 0;

struct ioroCtx_t
{
public:
	void* m_ptr;
private:
	oroU32 m_storage;

public:
	oroApi getApi() const { return (oroApi)m_storage; }
	void setApi( oroApi api ) { m_storage = api; }
};

struct ioroDevice
{
private:
	oroU32 m_api : 4;
	oroU32 m_deviceIdx : 16;

public:
	ioroDevice( int src = 0)
	{
		((int*)this)[0] = src;
	}

	oroApi getApi() const { return (oroApi)m_api; }
	void setApi(oroApi api) { m_api = api; }
	int getDevice() const { return m_deviceIdx; }
	void setDevice( int d ) { m_deviceIdx = d; }
};

inline 
oroApi getRawDeviceIndex( int& deviceId ) 
{
	int n[2] = { 0, 0 };
	oroGetDeviceCount( &n[0], ORO_API_HIP );
	oroGetDeviceCount( &n[1], ORO_API_CUDA );

	oroApi api = deviceId < n[0] ? ORO_API_HIP : ORO_API_CUDA;
	if( api == ORO_API_CUDA )
		deviceId -= n[0];
	return api;
}

int oroInitialize( oroApi api, oroU32 flags )
{
	s_api = api;
	int e = 0;
	s_loadedApis = 0;
	if( api & ORO_API_CUDA )
	{
		e = cuewInit( CUEW_INIT_CUDA | CUEW_INIT_NVRTC );
		if( e == 0 )
			s_loadedApis |= ORO_API_CUDA;
	}
	if( api & ORO_API_HIP )
	{
		e = hipewInit( HIPEW_INIT_HIP );
		if( e == 0 )
			s_loadedApis |= ORO_API_HIP;
	}
	if( s_loadedApis == 0 )
		return ORO_ERROR_OPEN_FAILED;
	return ORO_SUCCESS;
}
oroApi oroGetCurAPI(oroU32 flags)
{
	return s_api;
}

void* getRawCtx( oroCtx ctx ) 
{ 
	ioroCtx_t* c = (ioroCtx_t*)ctx;
	return c->m_ptr;
}

oroDevice getRawDevice( oroDevice dev )
{
	ioroDevice d( dev );
	return d.getDevice();
}


//=================================

inline
oroError hip2oro( hipError_t a )
{
	return (oroError)a;
}
inline
oroError cu2oro( CUresult a )
{
	return (oroError)a;
}
inline
oroError cuda2oro(cudaError_t a)
{
	return (oroError)a;
}
inline
CUcontext* oroCtx2cu( oroCtx* a )
{
	ioroCtx_t* b = *a;
	return (CUcontext*)&b->m_ptr;
}
inline
hipCtx_t* oroCtx2hip( oroCtx* a )
{
	ioroCtx_t* b = *a;
	return (hipCtx_t*)&b->m_ptr;
}
inline
orortcResult hiprtc2oro( hiprtcResult a )
{
	return (orortcResult)a;
}
inline
orortcResult nvrtc2oro( nvrtcResult a )
{
	return (orortcResult)a;
}

#define __ORO_FUNC1( cuname, hipname ) if( s_api == ORO_API_CUDA ) return cu2oro( cu##cuname ); if( s_api == ORO_API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC1X( API, cuname, hipname ) if( API == ORO_API_CUDA ) return cu2oro( cu##cuname ); if( API == ORO_API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC2( cudaname, hipname ) if( s_api == ORO_API_CUDA ) return cuda2oro( cuda##cudaname ); if( s_api == ORO_API_HIP ) return hip2oro( hip##hipname );
//#define __ORO_FUNC1( cuname, hipname ) if( s_api == ORO_API_CUDA || API == ORO_API_CUDA ) return cu2oro( cu##cuname ); if( s_api == API_HIP || API == API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC( name ) if( s_api == ORO_API_CUDA ) return cu2oro( cu##name ); if( s_api == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORO_FUNCX( API, name ) if( API == ORO_API_CUDA ) return cu2oro( cu##name ); if( API == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORO_CTXT_FUNC( name ) __ORO_FUNC1(Ctx##name, name)
#define __ORO_CTXT_FUNCX( API, name ) __ORO_FUNC1X(API, Ctx##name, name)
//#define __ORO_CTXT_FUNC( name ) if( s_api == ORO_API_CUDA ) return cu2oro( cuCtx##name ); if( s_api == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORORTC_FUNC1( cuname, hipname ) if( s_api == ORO_API_CUDA ) return nvrtc2oro( nvrtc##cuname ); if( s_api == ORO_API_HIP ) return hiprtc2oro( hiprtc##hipname );


oroError OROAPI oroGetErrorName(oroError error, const char** pStr)
{
	__ORO_FUNC1(GetErrorName((CUresult)error, pStr),
		GetErrorName((hipError_t)error, pStr));
	return oroErrorUnknown;
}

oroError OROAPI oroGetErrorString(oroError error, const char** pStr)
{
	__ORO_FUNC1(GetErrorString((CUresult)error, pStr),
		GetErrorString((hipError_t)error, pStr));
	return oroErrorUnknown;
}

oroError OROAPI oroInit(unsigned int Flags)
{
	oroU32 e = 0;
	if( s_loadedApis & ORO_API_HIP )
		e |= hip2oro(hipInit(Flags) );
	if (s_loadedApis & ORO_API_CUDA)
		e |= cu2oro(cuInit(Flags));
	return (oroError)e;
}

oroError OROAPI oroDriverGetVersion(int* driverVersion)
{
	__ORO_FUNC( DriverGetVersion(driverVersion) );
	return oroErrorUnknown;
}

oroError OROAPI oroGetDevice(int* device)
{
	__ORO_CTXT_FUNC( GetDevice(device) );
	return oroErrorUnknown;
}

oroError OROAPI oroGetDeviceCount(int* count, oroApi iapi)
{
	oroU32 api = 0;
	if( iapi == ORO_API_AUTOMATIC )
		api = (ORO_API_HIP| ORO_API_CUDA);
	else
		api = iapi;

	*count = 0;
	oroU32 e = 0;
	if( (api & s_loadedApis) & ORO_API_HIP )
	{
		int c = 0;
		e = hip2oro(hipGetDeviceCount(&c));
		if( e == 0 )
			*count += c;
	}
	if( (api & s_loadedApis) & ORO_API_CUDA )
	{
		int c = 0;
		e = cu2oro(cuDeviceGetCount(&c));
		if( e == 0 )
			*count += c;
	}
	return oroSuccess;
}

oroError OROAPI oroGetDeviceProperties(oroDeviceProp* props, oroDevice dev)
{
	ioroDevice d( dev );
	int deviceId = d.getDevice();
	// get API from raw device index, dev
	oroApi api = getRawDeviceIndex(dev);
	if( api == ORO_API_HIP )
		return hip2oro(hipGetDeviceProperties((hipDeviceProp_t*)props, deviceId));
	if( api == ORO_API_CUDA )
	{
		cudaDeviceProp p;
		cudaError_t e = cudaGetDeviceProperties( &p, deviceId );
		if (e != CUDA_SUCCESS)
			return oroErrorUnknown;
		strcpy( props->name, p.name );
		strcpy( props->gcnArchName, "" );
		props->totalGlobalMem = p.totalGlobalMem;
		props->pciDomainID = p.pciDomainID;
		props->pciBusID = p.pciBusID;
		props->pciDeviceID = p.pciDeviceID;
		memcpy(props->maxThreadsDim, p.maxThreadsDim, 3*sizeof(int));
		memcpy(props->maxGridSize, p.maxGridSize, 3*sizeof(int));
		props->maxThreadsPerBlock = p.maxThreadsPerBlock;
		return oroSuccess;
	}
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceGet(oroDevice* device, int ordinal )
{
	oroApi api = getRawDeviceIndex( ordinal );

	ioroDevice d;
	if (api == ORO_API_HIP)
	{
		int t;
		auto e = hipDeviceGet(&t, ordinal);
		d.setApi( api );
		d.setDevice( t );
		*(ioroDevice*)device = d;
		return hip2oro(e);
	}
	if (api == ORO_API_CUDA)
	{
		int t;
		auto e = cuDeviceGet(&t, ordinal);
		d.setApi(api);
		d.setDevice(t);
		*(ioroDevice*)device = d;
		return cu2oro(e);
	}
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceGetName(char* name, int len, oroDevice dev)
{
	ioroDevice d( dev );
	__ORO_FUNCX( d.getApi(), DeviceGetName(name, len, d.getDevice() ) );
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceGetAttribute(int* pi, oroDeviceAttribute attrib, oroDevice dev)
{
	ioroDevice d( dev );
	__ORO_FUNC1X( d.getApi(), DeviceGetAttribute( pi, (CUdevice_attribute)attrib, d.getDevice() ), DeviceGetAttribute( pi, (hipDeviceAttribute_t)attrib, d.getDevice() ) );
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceComputeCapability(int* major, int* minor, oroDevice dev)
{
	return oroErrorUnknown;
}

oroError OROAPI oroDevicePrimaryCtxRetain(oroCtx* pctx, oroDevice dev)
{
	return oroErrorUnknown;
}

oroError OROAPI oroDevicePrimaryCtxRelease(oroDevice dev)
{
	return oroErrorUnknown;
}

oroError OROAPI oroDevicePrimaryCtxSetFlags(oroDevice dev, unsigned int flags)
{
	return oroErrorUnknown;
}

oroError OROAPI oroDevicePrimaryCtxGetState(oroDevice dev, unsigned int* flags, int* active)
{
	return oroErrorUnknown;
}

oroError OROAPI oroDevicePrimaryCtxReset(oroDevice dev)
{
	return oroErrorUnknown;
}

oroError OROAPI oroCtxCreate(oroCtx* pctx, unsigned int flags, oroDevice dev)
{
	ioroDevice d( dev );
	ioroCtx_t* ctxt = new ioroCtx_t;
	ctxt->setApi( d.getApi() );
	(*pctx) = ctxt;
	s_api = ctxt->getApi();
	__ORO_FUNC1X( d.getApi(), CtxCreate( oroCtx2cu( pctx ), flags, d.getDevice() ), CtxCreate( oroCtx2hip( pctx ), flags, d.getDevice() ) );
	return oroErrorUnknown;
}

oroError OROAPI oroCtxDestroy(oroCtx ctx)
{
	int e = 0;
	if( s_api == ORO_API_CUDA ) e = cuCtxDestroy( *oroCtx2cu( &ctx ) );
	if( s_api == ORO_API_HIP ) e = hipCtxDestroy( *oroCtx2hip( &ctx ) );

	if( e )
		return oroErrorUnknown;
	ioroCtx_t* c = (ioroCtx_t*)ctx;
	delete c;
	return oroSuccess;
}
/*
oroError OROAPI oroCtxPushCurrent(oroCtx ctx);
oroError OROAPI oroCtxPopCurrent(oroCtx* pctx);
*/

oroError OROAPI oroCtxSetCurrent(oroCtx ctx)
{
	s_api = ctx->getApi();
	__ORO_FUNC1( CtxSetCurrent( *oroCtx2cu(&ctx) ), CtxSetCurrent( *oroCtx2hip(&ctx) ) );
	return oroErrorUnknown;
}

oroError OROAPI oroCtxGetCurrent(oroCtx* pctx)
{
	__ORO_FUNC1( CtxGetCurrent( oroCtx2cu(pctx) ), CtxGetCurrent( oroCtx2hip(pctx) ) );
	return oroErrorUnknown;
}
/*
oroError OROAPI oroCtxGetDevice(oroDevice* device);
oroError OROAPI oroCtxGetFlags(unsigned int* flags);
*/

oroError OROAPI oroCtxSynchronize(void)
{
	__ORO_FUNC( CtxSynchronize() );
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceSynchronize(void)
{
	__ORO_FUNC1( CtxSynchronize(), DeviceSynchronize() );
	return oroErrorUnknown;
}

//oroError OROAPI oroCtxGetCacheConfig(hipFuncCache_t* pconfig);
//oroError OROAPI oroCtxSetCacheConfig(hipFuncCache_t config);
//oroError OROAPI oroCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
//oroError OROAPI oroCtxSetSharedMemConfig(hipSharedMemConfig config);
oroError OROAPI oroCtxGetApiVersion(oroCtx ctx, unsigned int* version)
{
	__ORO_FUNC1( CtxGetApiVersion(*oroCtx2cu(&ctx), version ), CtxGetApiVersion(*oroCtx2hip(&ctx), version ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleLoad(oroModule* module, const char* fname)
{
	__ORO_FUNC1( ModuleLoad( (CUmodule*)module, fname ), ModuleLoad( (hipModule_t*)module, fname ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleLoadData(oroModule* module, const void* image)
{
	__ORO_FUNC1( ModuleLoadData( (CUmodule*)module, image ), ModuleLoadData( (hipModule_t*)module, image ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleLoadDataEx(oroModule* module, const void* image, unsigned int numOptions, oroJitOption* options, void** optionValues)
{
	__ORO_FUNC1( ModuleLoadDataEx( (CUmodule*)module, image, numOptions, (CUjit_option*)options, optionValues ),
		ModuleLoadDataEx( (hipModule_t*)module, image, numOptions, (hipJitOption*)options, optionValues ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleUnload(oroModule module)
{
	__ORO_FUNC1( ModuleUnload( (CUmodule)module ), ModuleUnload( (hipModule_t)module ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleGetFunction(oroFunction* hfunc, oroModule hmod, const char* name)
{
	__ORO_FUNC1( ModuleGetFunction( (CUfunction*)hfunc, (CUmodule)hmod, name ), 
		ModuleGetFunction( (hipFunction_t*)hfunc, (hipModule_t)hmod, name ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleGetGlobal(oroDeviceptr* dptr, size_t* bytes, oroModule hmod, const char* name)
{
	__ORO_FUNC1( ModuleGetGlobal( (CUdeviceptr*)dptr, bytes, (CUmodule)hmod, name ), 
		ModuleGetGlobal( (hipDeviceptr_t*)dptr, bytes, (hipModule_t)hmod, name ) );
	return oroErrorUnknown;
}
//oroError OROAPI oroModuleGetTexRef(textureReference** pTexRef, oroModule hmod, const char* name);
oroError OROAPI oroMemGetInfo(size_t* free, size_t* total)
{
	return oroErrorUnknown;
}
oroError OROAPI oroMalloc(oroDeviceptr* dptr, size_t bytesize)
{
	__ORO_FUNC1( MemAlloc((CUdeviceptr*)dptr, bytesize), Malloc( dptr, bytesize ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMalloc2(oroDeviceptr* dptr, size_t bytesize)
{
	__ORO_FUNC2( Malloc((CUdeviceptr*)dptr, bytesize), Malloc(dptr, bytesize) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemAllocPitch(oroDeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
	return oroErrorUnknown;
}
oroError OROAPI oroFree(oroDeviceptr dptr)
{
	__ORO_FUNC1( MemFree( dptr ), Free( dptr ) );
	return oroErrorUnknown;
}
oroError OROAPI oroFree2(oroDeviceptr dptr)
{
	__ORO_FUNC2( Free((CUdeviceptr)dptr), Free(dptr) );
	return oroErrorUnknown;
}

//-------------------
oroError OROAPI oroMemcpy(void *dstDevice, void* srcHost, size_t ByteCount, oroMemcpyKind kind)
{
	__ORO_FUNC2( Memcpy(dstDevice, srcHost, ByteCount, (cudaMemcpyKind)kind),
		Memcpy(dstDevice, srcHost, ByteCount, (hipMemcpyKind)kind) );
	return oroErrorUnknown;
}

oroError OROAPI oroMemcpyHtoD(oroDeviceptr dstDevice, void* srcHost, size_t ByteCount)
{
	__ORO_FUNC1( MemcpyHtoD( dstDevice, srcHost, ByteCount ),
		MemcpyHtoD( dstDevice, srcHost, ByteCount ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoH(void* dstHost, oroDeviceptr srcDevice, size_t ByteCount)
{
	__ORO_FUNC1( MemcpyDtoH( dstHost, srcDevice, ByteCount ),
		MemcpyDtoH( dstHost, srcDevice, ByteCount ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoD(oroDeviceptr dstDevice, oroDeviceptr srcDevice, size_t ByteCount)
{
	__ORO_FUNC( MemcpyDtoD( dstDevice, srcDevice, ByteCount ) );
	return oroErrorUnknown;
}

oroError OROAPI oroMemset(oroDeviceptr dstDevice, unsigned int ui, size_t N)
{
	__ORO_FUNC1( MemsetD8( (CUdeviceptr)dstDevice, ui, N ), Memset((void*)dstDevice, ui, N));
	return oroErrorUnknown;
}

oroError OROAPI oroMemsetD8(oroDeviceptr dstDevice, unsigned char ui, size_t N)
{
	__ORO_FUNC(MemsetD8(dstDevice, ui, N));
	return oroErrorUnknown;
}
oroError OROAPI oroMemsetD16(oroDeviceptr dstDevice, unsigned short ui, size_t N)
{
	__ORO_FUNC(MemsetD16(dstDevice, ui, N));
	return oroErrorUnknown;
}
oroError OROAPI oroMemsetD32(oroDeviceptr dstDevice, unsigned int ui, size_t N)
{
	__ORO_FUNC(MemsetD32(dstDevice, ui, N));
	return oroErrorUnknown;
}

//-------------------
oroError OROAPI oroModuleLaunchKernel(oroFunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, oroStream hStream, void** kernelParams, void** extra)
{
	__ORO_FUNC1( LaunchKernel( (CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra ),
		ModuleLaunchKernel( (hipFunction_t)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (hipStream_t)hStream, kernelParams, extra ) );
	return oroErrorUnknown;
}

//-------------------
oroError OROAPI oroImportExternalMemory(oroExternalMemory_t* extMem_out, const oroExternalMemoryHandleDesc* memHandleDesc)
{
	__ORO_FUNC2( ImportExternalMemory( (cudaExternalMemory_t*)extMem_out, (const cudaExternalMemoryHandleDesc*)memHandleDesc ),
		ImportExternalMemory( (hipExternalMemory_t*)extMem_out, (const hipExternalMemoryHandleDesc*)memHandleDesc ) );
	return oroErrorUnknown;
}
//-------------------
oroError OROAPI oroExternalMemoryGetMappedBuffer(void **devPtr, oroExternalMemory_t extMem, const oroExternalMemoryBufferDesc* bufferDesc)
{
	__ORO_FUNC2( ExternalMemoryGetMappedBuffer( devPtr, (cudaExternalMemory_t)extMem, (const cudaExternalMemoryBufferDesc*)bufferDesc ),
		ExternalMemoryGetMappedBuffer( devPtr, (hipExternalMemory_t)extMem, (const hipExternalMemoryBufferDesc*)bufferDesc ) );
	return oroErrorUnknown;
}
//-------------------
oroError OROAPI oroDestroyExternalMemory(oroExternalMemory_t extMem)
{
	__ORO_FUNC2( DestroyExternalMemory( (cudaExternalMemory_t)extMem ),
		DestroyExternalMemory( (hipExternalMemory_t)extMem ) );
	return oroErrorUnknown;
}
oroError OROAPI oroGetLastError(oroError oro_error)
{
	__ORO_FUNC2(GetLastError((cudaError_t)oro_error),
		GetLastError((hipError_t)oro_error));
	return oroErrorUnknown;
}
//-------------------
orortcResult OROAPI orortcGetErrorString(orortcResult result)
{
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcAddNameExpression(orortcProgram prog, const char* name_expression)
{
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCompileProgram(orortcProgram prog, int numOptions, const char** options)
{
	__ORORTC_FUNC1( CompileProgram( (nvrtcProgram)prog, numOptions, options ),
		CompileProgram( (hiprtcProgram)prog, numOptions, options ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCreateProgram(orortcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames)
{
	__ORORTC_FUNC1( CreateProgram( (nvrtcProgram*)prog, src, name, numHeaders, headers, includeNames ), 
		CreateProgram( (hiprtcProgram*)prog, src, name, numHeaders, headers, includeNames ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcDestroyProgram(orortcProgram* prog)
{
	__ORORTC_FUNC1( DestroyProgram( (nvrtcProgram*)prog), 
		DestroyProgram( (hiprtcProgram*)prog ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetLoweredName(orortcProgram prog, const char* name_expression, const char** lowered_name)
{
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetProgramLog(orortcProgram prog, char* log)
{
	__ORORTC_FUNC1( GetProgramLog( (nvrtcProgram)prog, log ), 
		GetProgramLog( (hiprtcProgram)prog, log ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetProgramLogSize(orortcProgram prog, size_t* logSizeRet)
{
	__ORORTC_FUNC1( GetProgramLogSize( (nvrtcProgram)prog, logSizeRet), 
		GetProgramLogSize( (hiprtcProgram)prog, logSizeRet ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCode(orortcProgram prog, char* code)
{
	__ORORTC_FUNC1( GetPTX( (nvrtcProgram)prog, code ), 
		GetCode( (hiprtcProgram)prog, code ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCodeSize(orortcProgram prog, size_t* codeSizeRet)
{
	__ORORTC_FUNC1( GetPTXSize( (nvrtcProgram)prog, codeSizeRet ), 
		GetCodeSize( (hiprtcProgram)prog, codeSizeRet ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}

// Implementation of oroPointerGetAttributes is hacky due to differences between CUDA and HIP
oroError OROAPI oroPointerGetAttributes(oroPointerAttribute* attr, oroDeviceptr dptr)
{
	if (s_api == ORO_API_CUDA)
	{
		unsigned int data;
		return cu2oro(cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr));
	}
	if (s_api == ORO_API_HIP) 
		return hip2oro(hipPointerGetAttributes((hipPointerAttribute_t*)attr, (void*)dptr));

	return oroErrorUnknown;
}

//-----------------
oroError OROAPI oroStreamCreate(oroStream* stream)
{
	__ORO_FUNC2(StreamCreate((cudaStream_t*)stream),
		StreamCreate((hipStream_t*)stream));

	return oroErrorUnknown;
}


