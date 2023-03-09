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
#include <unordered_map>
#include <mutex>

#if defined( ORO_INTEL )
#include <ze_api.h>
#include <fstream>
//https://spec.oneapi.io/level-zero/latest/api.html

const char* getTempFilePath() { return "__x.cl"; }
const char* getCompiledFilePath() { return "__x_Gen12LPdg1.spv"; }
std::string getCompiledPathPostfix() { return "_Gen12LPdg1.spv"; }
#endif

std::unordered_map<void*, oroCtx> s_oroCtxs;
static std::mutex mtx;
thread_local static oroApi s_api = ORO_API_HIP;
static oroU32 s_loadedApis = 0;
#if defined( ORO_INTEL )
static ze_driver_handle_t s_driver = 0;

struct IntelContext
{
	ze_context_handle_t m_ctxt = 0;
	ze_device_handle_t m_device = 0;
	ze_command_list_handle_t m_cmdList = 0;
	ze_command_queue_handle_t m_queue = 0;
};

static IntelContext s_iCtxt;
#endif



static void todo(const char* txt) 
{ 
	printf( "todo: %s\n", txt );
}

#define TODO(x) todo( x );

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
	oroU64 m_api : 6;
	oroU64 m_deviceIdx : 57;

public:
	ioroDevice( oroU64 src = 0)
	{
		((oroU64*)this)[0] = src;
	}

	oroApi getApi() const { return (oroApi)m_api; }
	void setApi(oroApi api) { m_api = api; }
	oroU64 getDevice() const { return m_deviceIdx; }
	void setDevice( oroU64 d ) { m_deviceIdx = d; }
};

inline 
oroApi getRawDeviceIndex( int& deviceId ) 
{
	int n[3] = { 0, 0, 0 };
	oroGetDeviceCount( &n[0], ORO_API_HIP );
	oroGetDeviceCount( &n[1], ORO_API_CUDADRIVER );
	oroGetDeviceCount( &n[2], ORO_API_INTEL );

	oroApi api = ORO_API_HIP;
	if( deviceId < n[0] )
		api = ORO_API_HIP;
	else if( deviceId < n[0] + n[1] )
		api = ORO_API_CUDADRIVER;
	else
		api = ORO_API_INTEL;

	if( api & ORO_API_CUDADRIVER )
		deviceId -= n[0];
#if defined( ORO_INTEL )
	if( api & ORO_API_INTEL ) 
		deviceId -= n[0] + n[1];
#endif
	return api;
}

int oroInitialize( oroApi api, oroU32 flags )
{
	s_api = api;
	int e = 0;
	s_loadedApis = 0;

	if( api & ORO_API_CUDA )
	{
		cuuint32_t flag = 0;
		if( api & ORO_API_CUDADRIVER )
		{
			flag |= CUEW_INIT_CUDA;
		}
		if( api & ORO_API_CUDARTC )
		{
			flag |= CUEW_INIT_NVRTC;
		}
		
		int resultDriver, resultRtc;
		cuewInit( &resultDriver, &resultRtc, flag );

		if( resultDriver == CUEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_CUDADRIVER;
		}
		if( resultRtc == CUEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_CUDARTC;
		}
	}
	if( api & ORO_API_HIP )
	{
		hipuint32_t flag = 0;
		if( api & ORO_API_HIPDRIVER )
		{
			flag |= HIPEW_INIT_HIPDRIVER;
		}
		if( api & ORO_API_HIPRTC )
		{
			flag |= HIPEW_INIT_HIPRTC;
		}

		int resultDriver, resultRtc;
		hipewInit( &resultDriver, &resultRtc, flag );

		if( resultDriver == HIPEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_HIPDRIVER;
		}
		if( resultRtc == HIPEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_HIPRTC;
		}
	}
#if defined( ORO_INTEL )
	if( api & ORO_API_INTEL )
	{
		ze_result_t e;
		e = zeInit( 0 );
		if( e == ZE_RESULT_SUCCESS )
		{
			s_loadedApis |= ORO_API_INTEL;
		}
	}
#endif
	if( s_loadedApis == 0 )
		return ORO_ERROR_OPEN_FAILED;
	return ORO_SUCCESS;
}
oroApi oroLoadedAPI() 
{
	return (oroApi)s_loadedApis;
}
oroApi oroGetCurAPI(oroU32 flags)
{
	return s_api;
}

void* oroGetRawCtx( oroCtx ctx ) 
{ 
	ioroCtx_t* c = (ioroCtx_t*)ctx;
	return c->m_ptr;
}

//oroCtx setRawCtx( oroApi api, void* ctx )
oroError oroCtxCreateFromRaw( oroCtx* ctxOut, oroApi api, void* ctxIn )
{ 
	ioroCtx_t* c = new ioroCtx_t;
	c->m_ptr = ctxIn;
	c->setApi( api );
	*ctxOut = c;
	return oroSuccess;
}

oroError oroCtxCreateFromRawDestroy( oroCtx ctx ) 
{
	ioroCtx_t* c = (ioroCtx_t*)ctx;
	delete c;
	return oroSuccess;
}

oroDevice oroGetRawDevice( oroDevice dev )
{
	ioroDevice d( dev );
	return d.getDevice();
}

oroDevice oroSetRawDevice( oroApi api, oroDevice dev ) 
{
	ioroDevice d( dev );
	d.setApi( api );
	return *(oroDevice*)&d;
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
#if defined( ORO_INTEL )
inline ze_context_handle_t* oroCtx2intel( oroCtx* a )
{
	ioroCtx_t* b = *a;
	return (ze_context_handle_t*)&b->m_ptr;
}
inline ze_device_handle_t oroDevice2intel( oroU64 d ) 
{ 
	return (ze_device_handle_t)d;
}
#endif
inline orortcResult hiprtc2oro( hiprtcResult a ) {
	return (orortcResult)a;
}
inline
orortcResult nvrtc2oro( nvrtcResult a )
{

	return (orortcResult)a;
}

inline orortcResult cu2orortc( CUresult a ) { return (orortcResult)a; }


#define __ORO_FUNC1( cuname, hipname ) if( s_api & ORO_API_CUDADRIVER ) return cu2oro( cu##cuname ); if( s_api == ORO_API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC1X( API, cuname, hipname ) if( API & ORO_API_CUDADRIVER ) return cu2oro( cu##cuname ); if( API == ORO_API_HIP ) return hip2oro( hip##hipname );
//#define __ORO_FUNC2( cudaname, hipname ) if( s_api == ORO_API_CUDA ) return cuda2oro( cuda##cudaname ); if( s_api == ORO_API_HIP ) return hip2oro( hip##hipname );
//#define __ORO_FUNC1( cuname, hipname ) if( s_api == ORO_API_CUDA || API == ORO_API_CUDA ) return cu2oro( cu##cuname ); if( s_api == API_HIP || API == API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC( name ) if( s_api & ORO_API_CUDADRIVER ) return cu2oro( cu##name ); if( s_api == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORO_FUNCX( API, name ) if( API & ORO_API_CUDADRIVER ) return cu2oro( cu##name ); if( API == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORO_CTXT_FUNC( name ) __ORO_FUNC1(Ctx##name, name)
#define __ORO_CTXT_FUNCX( API, name ) __ORO_FUNC1X(API, Ctx##name, name)
//#define __ORO_CTXT_FUNC( name ) if( s_api == ORO_API_CUDA ) return cu2oro( cuCtx##name ); if( s_api == ORO_API_HIP ) return hip2oro( hip##name );
#define __ORORTC_FUNC1( cuname, hipname ) if( s_api & ORO_API_CUDADRIVER ) return nvrtc2oro( nvrtc##cuname ); if( s_api == ORO_API_HIP ) return hiprtc2oro( hiprtc##hipname );
#define __ORO_RET_ERR( e ) if( s_api & ORO_API_CUDADRIVER ) return cu2oro((CUresult)e ); if( s_api == ORO_API_HIP ) return hip2oro( (hipError_t)e );


oroError OROAPI oroGetErrorName(oroError error, const char** pStr)
{
	__ORO_FUNC1(GetErrorName((CUresult)error, pStr),
		GetErrorName((hipError_t)error, pStr));
	return oroErrorUnknown;
}

oroError OROAPI oroGetErrorString( oroError error, const char** pStr )
{
	if( s_api & ORO_API_CUDADRIVER ) return cu2oro(cuGetErrorString( (CUresult)error, pStr ));
	else
	{
		*pStr = hipGetErrorString( (hipError_t)error );
		if (*pStr) return oroSuccess;
	}
	return oroErrorUnknown;
}

oroError OROAPI oroInit(unsigned int Flags)
{
	oroU32 e0 = 0;
	oroU32 e1 = 0;
	if( s_loadedApis & ORO_API_HIP )
	{
		e0 = hip2oro( hipInit( Flags ) );
	}
	if (s_loadedApis & ORO_API_CUDADRIVER)
	{
		e1 = cu2oro( cuInit( Flags ) );
	}
#if defined( ORO_INTEL )
	if( s_loadedApis & ORO_API_INTEL )
	{
		uint32_t driverCount = 0;
		auto e = zeDriverGet( &driverCount, nullptr );
		if( e == ZE_RESULT_SUCCESS )
		{
			std::vector<ze_driver_handle_t> drivers( driverCount );
			e = zeDriverGet( &driverCount, drivers.data() );

			if( e != ZE_RESULT_SUCCESS ) return oroErrorUnknown;
			if( driverCount > 0 )
				s_driver = drivers[0];
		}
		else
			return oroErrorUnknown;
	}
#endif
	return ( e0 == 0 || e1 == 0 ) ? oroSuccess : oroErrorUnknown;
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
		api = (ORO_API_HIP|ORO_API_CUDADRIVER);
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
	if( (api & s_loadedApis) & (ORO_API_CUDADRIVER) )
	{
		int c = 0;
		e = cu2oro(cuDeviceGetCount(&c));
		if( e == 0 )
			*count += c;
	}
#if defined( ORO_INTEL )
	if( ( api & s_loadedApis ) & ORO_API_INTEL )
	{
		uint32_t c = 0;
		zeDeviceGet( s_driver, &c, nullptr );
		*count += c;
	}
#endif
	return oroSuccess;
}

oroError OROAPI oroGetDeviceProperties(oroDeviceProp* props, oroDevice dev)
{
	ioroDevice d( dev );
	oroU64 deviceId = d.getDevice();
	oroApi api = d.getApi();
	if( api == ORO_API_HIP )
		return hip2oro(hipGetDeviceProperties((hipDeviceProp_t*)props, deviceId));
	if( api & ORO_API_CUDADRIVER )
	{
		CUdevprop p;
		CUresult e = cuDeviceGetProperties(&p, deviceId);
		e = cuDeviceGetName(props->name, 256, deviceId);
		strcpy( props->gcnArchName, "" );
		memcpy( props->maxThreadsDim, p.maxThreadsDim, 3 * sizeof( int ) );
		memcpy( props->maxGridSize, p.maxGridSize, 3 * sizeof( int ) );
		props->maxThreadsPerBlock = p.maxThreadsPerBlock;
		props->sharedMemPerBlock = p.sharedMemPerBlock;
		props->totalConstMem = p.totalConstantMemory;
		props->regsPerBlock = p.regsPerBlock;
		props->memPitch = p.memPitch;
		props->clockRate = p.clockRate;

		cu2oro( cuDeviceTotalMem( &props->totalGlobalMem, deviceId ) );

		e = cuDeviceGetAttribute( &props->pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, deviceId );
		e = cuDeviceGetAttribute(&props->pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, deviceId);
		e = cuDeviceGetAttribute(&props->pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, deviceId);
		e = cuDeviceGetAttribute(&props->multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceId);
		e = cuDeviceGetAttribute( &props->warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, deviceId );
		e = cuDeviceGetAttribute( (int*)&props->textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, deviceId );
		e = cuDeviceGetAttribute( &props->kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, deviceId );
		e = cuDeviceGetAttribute( &props->integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, deviceId );
		e = cuDeviceGetAttribute( &props->canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, deviceId );
		e = cuDeviceGetAttribute( &props->computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, deviceId );
		e = cuDeviceGetAttribute( &props->concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, deviceId );
		e = cuDeviceGetAttribute( &props->ECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, deviceId );
		e = cuDeviceGetAttribute( &props->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceId );
		e = cuDeviceGetAttribute( &props->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceId );

		return oroSuccess;
	}
#if defined( ORO_INTEL )
	if( api & ORO_API_INTEL )
	{
		ze_device_properties_t p = {};
		p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
		zeDeviceGetProperties( (ze_device_handle_t)deviceId, &p );
		strcpy( props->name, p.name );
		strcpy( props->gcnArchName, "" );
		props->multiProcessorCount = p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice;

		return oroSuccess;
	}
#endif
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
	if (api & ORO_API_CUDADRIVER)
	{
		int t;
		auto e = cuDeviceGet(&t, ordinal);
		d.setApi(api);
		d.setDevice(t);
		*(ioroDevice*)device = d;
		return cu2oro(e);
	}
#if defined( ORO_INTEL )
	if( api & ORO_API_INTEL )
	{
		uint32_t deviceCount = 0;
		zeDeviceGet( s_driver, &deviceCount, nullptr );
		std::vector<ze_device_handle_t> devices( deviceCount );
		zeDeviceGet( s_driver, &deviceCount, devices.data() );

		auto dd = devices[ordinal];
		ze_device_properties_t device_properties = {};
		device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
		zeDeviceGetProperties( dd, &device_properties );
		if( device_properties.type == ZE_DEVICE_TYPE_GPU )
		{
			d.setApi( api );
			d.setDevice( (oroU64)dd );
			*(ioroDevice*)device = d;
			s_iCtxt.m_device = dd;
			return oroSuccess;
		}
	}
#endif
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

oroError OROAPI oroDeviceGetLimit( size_t* pValue, oroLimit limit )
{
	__ORO_FUNC1( CtxGetLimit( pValue, (CUlimit)limit ), DeviceGetLimit( pValue, (hipLimit_t)limit ) );
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceSetLimit( oroLimit limit, size_t value )
{
	__ORO_FUNC1( CtxSetLimit( (CUlimit)limit, value ), DeviceSetLimit( (hipLimit_t)limit, value ) );
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
	int e = oroErrorUnknown;
	if( s_api & ORO_API_CUDADRIVER ) e = cuCtxCreate( oroCtx2cu( pctx ), flags, d.getDevice() );
	if( s_api == ORO_API_HIP ) e = hipCtxCreate( oroCtx2hip( pctx ), flags, d.getDevice() );
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		ze_context_handle_t context;
		ze_context_desc_t context_desc = {};
		context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
		e = zeContextCreate( s_driver, &context_desc, oroCtx2intel( pctx ) );
		s_iCtxt.m_ctxt = *oroCtx2intel( pctx );
	}
#endif
	if( e )
	{
		__ORO_RET_ERR( e )
	}
	std::lock_guard<std::mutex> lock( mtx );
	s_oroCtxs[ctxt->m_ptr] = ctxt;
	return oroSuccess;
}

oroError OROAPI oroCtxDestroy(oroCtx ctx)
{
	std::lock_guard<std::mutex> lock( mtx );
	s_oroCtxs.erase( ctx->m_ptr );

#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeContextDestroy( s_iCtxt.m_ctxt );
		if( e != ZE_RESULT_SUCCESS ) return oroErrorUnknown;

		return oroSuccess;
	}
#endif
	int e = 0;
	if( s_api & ORO_API_CUDADRIVER ) e = cuCtxDestroy( *oroCtx2cu( &ctx ) );
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
	ioroCtx_t* ctxt = new ioroCtx_t;
	int e = oroErrorUnknown;
	if( s_api & ORO_API_CUDADRIVER ) e = cuCtxGetCurrent( oroCtx2cu( &ctxt ) );
	if( s_api == ORO_API_HIP ) e = hipCtxGetCurrent( oroCtx2hip( &ctxt ) );
	if( e )
	{
		__ORO_RET_ERR( e )
	}

	( *pctx ) = s_oroCtxs[ctxt->m_ptr];

	delete ctxt;
	return oroSuccess;
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
oroError OROAPI oroModuleLoadData( oroModule* module, const void* image, unsigned int imageSize )
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		if( imageSize == 0 ) return oroErrorUnknown;
		ze_module_desc_t moduleDesc = {};
		ze_module_build_log_handle_t buildLog;
		moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
		moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>( image );
		moduleDesc.inputSize = imageSize;
		moduleDesc.pBuildFlags = "";

		auto status = zeModuleCreate( s_iCtxt.m_ctxt, s_iCtxt.m_device, &moduleDesc, (ze_module_handle_t*)module, &buildLog );
		zeModuleBuildLogDestroy( buildLog );
		if( status == ZE_RESULT_SUCCESS ) return oroSuccess;

		return oroErrorUnknown;
	}
#endif
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
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeModuleDestroy( (ze_module_handle_t)module );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( ModuleUnload( (CUmodule)module ), ModuleUnload( (hipModule_t)module ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleGetFunction(oroFunction* hfunc, oroModule hmod, const char* name)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		ze_kernel_desc_t kernelDesc = {};
		kernelDesc.pKernelName = name;
		auto e = zeKernelCreate( (ze_module_handle_t)hmod, &kernelDesc, (ze_kernel_handle_t*)hfunc );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( ModuleGetFunction( (CUfunction*)hfunc, (CUmodule)hmod, name ), 
		ModuleGetFunction( (hipFunction_t*)hfunc, (hipModule_t)hmod, name ) );
	return oroErrorUnknown;
}
oroError OROAPI oroModuleGetGlobal(oroDeviceptr* dptr, size_t* bytes, oroModule hmod, const char* name)
{
	__ORO_FUNC1( ModuleGetGlobal( (CUdeviceptr*)dptr, bytes, (CUmodule)hmod, name ), 
		ModuleGetGlobal( (oroDeviceptr*)dptr, bytes, (hipModule_t)hmod, name ) );
	return oroErrorUnknown;
}
//oroError OROAPI oroModuleGetTexRef(textureReference** pTexRef, oroModule hmod, const char* name);
oroError OROAPI oroMemGetInfo(size_t* free, size_t* total)
{
	return oroErrorUnknown;
}
oroError OROAPI oroMalloc(oroDeviceptr* dptr, size_t bytesize)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		ze_device_mem_alloc_desc_t desc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
		desc.ordinal = 0;
		auto e = zeMemAllocDevice( s_iCtxt.m_ctxt, &desc, bytesize, 16, s_iCtxt.m_device, (void**)dptr );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( MemAlloc((CUdeviceptr*)dptr, bytesize), Malloc( dptr, bytesize ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMalloc2(oroDeviceptr* dptr, size_t bytesize)
{
	__ORO_FUNC1( MemAlloc((CUdeviceptr*)dptr, bytesize), Malloc(dptr, bytesize) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemAllocPitch(oroDeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
	return oroErrorUnknown;
}
oroError OROAPI oroFree(oroDeviceptr dptr)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeMemFree( s_iCtxt.m_ctxt, (void*)dptr );
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( MemFree( dptr ), Free( dptr ) );
	return oroErrorUnknown;
}
oroError OROAPI oroFree2(oroDeviceptr dptr)
{
	__ORO_FUNC1( MemFree((CUdeviceptr)dptr), Free(dptr) );
	return oroErrorUnknown;
}
oroError OROAPI oroHostRegister(void* p, size_t bytesize, unsigned int Flags)
{
	__ORO_FUNC1( MemHostRegister(p, bytesize, Flags), HostRegister(p, bytesize, Flags) );
	return oroErrorUnknown;
}
oroError OROAPI oroHostGetDevicePointer(oroDeviceptr* pdptr, void* p, unsigned int Flags)
{
	__ORO_FUNC1( MemHostGetDevicePointer((CUdeviceptr*)pdptr, p, Flags), HostGetDevicePointer(pdptr, p, Flags) );
	return oroErrorUnknown;
}
oroError OROAPI oroHostUnregister(void* p)
{
	__ORO_FUNC1( MemHostUnregister(p), HostUnregister(p) );
	return oroErrorUnknown;
}

//-------------------
/* oroError OROAPI oroMemcpy(void *dstDevice, void* srcHost, size_t ByteCount, oroMemcpyKind kind)
{
	__ORO_FUNC2( Memcpy(dstDevice, srcHost, ByteCount, (cudaMemcpyKind)kind),
		Memcpy(dstDevice, srcHost, ByteCount, (hipMemcpyKind)kind) );
	return oroErrorUnknown;
} */

oroError OROAPI oroMemcpyHtoD(oroDeviceptr dstDevice, void* srcHost, size_t ByteCount)
{
	__ORO_FUNC1( MemcpyHtoD( dstDevice, srcHost, ByteCount ),
		MemcpyHtoD( dstDevice, srcHost, ByteCount ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoH(void* dstHost, oroDeviceptr srcDevice, size_t ByteCount)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeCommandListAppendMemoryCopy( s_iCtxt.m_cmdList, dstHost, (const void*)srcDevice, ByteCount, 0, 0, 0 );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = (ze_result_t)oroStreamSynchronize( (oroStream)s_iCtxt.m_queue );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( MemcpyDtoH( dstHost, srcDevice, ByteCount ),
		MemcpyDtoH( dstHost, srcDevice, ByteCount ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoD(oroDeviceptr dstDevice, oroDeviceptr srcDevice, size_t ByteCount)
{
	__ORO_FUNC( MemcpyDtoD( dstDevice, srcDevice, ByteCount ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyHtoDAsync(oroDeviceptr dstDevice, const void* srcHost, size_t ByteCount, oroStream hStream)
{
	__ORO_FUNC1( MemcpyHtoDAsync( (CUdeviceptr)dstDevice, srcHost, ByteCount, (CUstream)hStream ), 
		MemcpyHtoDAsync( dstDevice, srcHost, ByteCount, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoHAsync(void* dstHost, oroDeviceptr srcDevice, size_t ByteCount, oroStream hStream) 
{
	__ORO_FUNC1( MemcpyDtoHAsync( dstHost, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream ), 
		MemcpyDtoHAsync( dstHost, (CUdeviceptr)srcDevice, ByteCount, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemcpyDtoDAsync( oroDeviceptr dstDevice, oroDeviceptr srcDevice, size_t ByteCount, oroStream hStream )
{
	__ORO_FUNC1( MemcpyDtoDAsync( dstDevice, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream ), 
		MemcpyDtoDAsync( dstDevice, srcDevice, ByteCount, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemset(oroDeviceptr dstDevice, unsigned int ui, size_t N)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeCommandListAppendMemoryFill( s_iCtxt.m_cmdList, (void*)dstDevice, &ui, sizeof( int ), N, 0, 0, 0 );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = (ze_result_t)oroStreamSynchronize( (oroStream)s_iCtxt.m_queue );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( MemsetD32( (CUdeviceptr)dstDevice, ui, N / sizeof( int ) ), MemsetD32( dstDevice, ui, N / sizeof(int) ) );
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
oroError OROAPI oroMemsetD8Async(oroDeviceptr dstDevice, unsigned char uc, size_t N, oroStream hStream) 
{ 
	__ORO_FUNC1( MemsetD8Async( dstDevice, uc, N, (CUstream)hStream ), MemsetD8Async( dstDevice, uc, N, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemsetD16Async(oroDeviceptr dstDevice, unsigned short us, size_t N, oroStream hStream) 
{
	__ORO_FUNC1( MemsetD16Async( dstDevice, us, N, (CUstream)hStream ), MemsetD16Async( dstDevice, us, N, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}
oroError OROAPI oroMemsetD32Async(oroDeviceptr dstDevice, unsigned int ui, size_t N, oroStream hStream) 
{
	__ORO_FUNC1( MemsetD32Async( dstDevice, ui, N, (CUstream)hStream ), MemsetD32Async( dstDevice, ui, N, (hipStream_t)hStream ) );
	return oroErrorUnknown;
}

//-------------------
oroError OROAPI oroFuncGetAttribute( int* pi, oroFunction_attribute attrib, oroFunction hfunc ) 
{
	__ORO_FUNC1( FuncGetAttribute( pi, (CUfunction_attribute)attrib, (CUfunction)hfunc ), FuncGetAttribute( pi, (hipFunction_attribute)attrib, (hipFunction_t)hfunc ) );
	return oroErrorUnknown;
}

oroError OROAPI oroModuleLaunchKernel(oroFunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, oroStream hStream, void** kernelParams, void** extra)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		ze_kernel_handle_t kernel = (ze_kernel_handle_t)f;

		ze_kernel_properties_t props = {};
		auto e = zeKernelGetProperties( kernel, &props );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;

		e = zeKernelSetGroupSize( kernel, blockDimX, blockDimY, blockDimZ );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		for(int i=0; i<props.numKernelArgs; i++)
		{
			e = zeKernelSetArgumentValue( kernel, i, sizeof( kernelParams[i] ), kernelParams[i] );
			if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		}
		ze_group_count_t dispatch;
		dispatch.groupCountX = gridDimX;
		dispatch.groupCountY = gridDimY;
		dispatch.groupCountZ = gridDimZ;
		e = zeCommandListAppendLaunchKernel( s_iCtxt.m_cmdList, kernel, &dispatch, nullptr, 0, nullptr );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( LaunchKernel( (CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra ),
		ModuleLaunchKernel( (hipFunction_t)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (hipStream_t)hStream, kernelParams, extra ) );
	return oroErrorUnknown;
}

oroError OROAPI oroModuleOccupancyMaxPotentialBlockSize( int* minGridSize, int* blockSize, oroFunction func, size_t dynamicSMemSize, int blockSizeLimit ) 
{ 
	if( s_api & ORO_API_CUDADRIVER )
	{
//		CUoccupancyB2DSize blockSizeToDynamicSMemSize;
		return cu2oro( cuOccupancyMaxPotentialBlockSize( minGridSize, blockSize, (CUfunction)func, 0, dynamicSMemSize, blockSizeLimit ) );
	}
	else
		return hip2oro( hipModuleOccupancyMaxPotentialBlockSize( minGridSize, blockSize, (hipFunction_t)func, dynamicSMemSize, blockSizeLimit ) );
	return oroErrorUnknown;
}

//-------------------
oroError OROAPI oroImportExternalMemory(oroExternalMemory_t* extMem_out, const oroExternalMemoryHandleDesc* memHandleDesc)
{
	__ORO_FUNC1( ImportExternalMemory( (CUexternalMemory*)extMem_out, (const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*)memHandleDesc ),
		ImportExternalMemory( (hipExternalMemory_t*)extMem_out, (const hipExternalMemoryHandleDesc*)memHandleDesc ) );
	return oroErrorUnknown;
}
//-------------------
oroError OROAPI oroExternalMemoryGetMappedBuffer(void **devPtr, oroExternalMemory_t extMem, const oroExternalMemoryBufferDesc* bufferDesc)
{
	__ORO_FUNC1( ExternalMemoryGetMappedBuffer( (CUdeviceptr*)devPtr, (CUexternalMemory)extMem, (const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*)bufferDesc ),
		ExternalMemoryGetMappedBuffer( devPtr, (hipExternalMemory_t)extMem, (const hipExternalMemoryBufferDesc*)bufferDesc ) );
	return oroErrorUnknown;
}
//-------------------
oroError OROAPI oroDestroyExternalMemory(oroExternalMemory_t extMem)
{
	__ORO_FUNC1( DestroyExternalMemory( (CUexternalMemory)extMem ),
		DestroyExternalMemory( (hipExternalMemory_t)extMem ) );
	return oroErrorUnknown;
}
/* oroError OROAPI oroGetLastError(oroError oro_error)
{
	__ORO_FUNC2(GetLastError((cudaError_t)oro_error),
		GetLastError((hipError_t)oro_error));
	return oroErrorUnknown;
} */
//-------------------
const char* OROAPI orortcGetErrorString(orortcResult result)
{
	if( s_api & ORO_API_CUDADRIVER ) return nvrtcGetErrorString( (nvrtcResult)result );
	else return hiprtcGetErrorString( (hiprtcResult)result );
	return 0;
}
orortcResult OROAPI orortcAddNameExpression( orortcProgram prog, const char* name_expression )
{
	__ORORTC_FUNC1( AddNameExpression( (nvrtcProgram)prog, name_expression ), AddNameExpression( (hiprtcProgram)prog, name_expression ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCompileProgram(orortcProgram prog, int numOptions, const char** options)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		std::string cmd = "ocloc -file " + std::string( getTempFilePath() );
		cmd += " -device dg1";// -output " + std::string( getCompiledFilePath() );
		system( cmd.c_str() );
		return ORORTC_SUCCESS;
	}
#endif
	__ORORTC_FUNC1( CompileProgram( (nvrtcProgram)prog, numOptions, options ),
		CompileProgram( (hiprtcProgram)prog, numOptions, options ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCreateProgram(orortcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		std::ofstream file;
		file.open( getTempFilePath() );
		file << src;
		file.close();
		return ORORTC_SUCCESS;
	}
#endif
	__ORORTC_FUNC1( CreateProgram( (nvrtcProgram*)prog, src, name, numHeaders, headers, includeNames ), 
		CreateProgram( (hiprtcProgram*)prog, src, name, numHeaders, headers, includeNames ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcDestroyProgram(orortcProgram* prog)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		return ORORTC_SUCCESS;
	}
#endif
	__ORORTC_FUNC1( DestroyProgram( (nvrtcProgram*)prog), 
		DestroyProgram( (hiprtcProgram*)prog ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetLoweredName(orortcProgram prog, const char* name_expression, const char** lowered_name)
{
	__ORORTC_FUNC1( GetLoweredName( (nvrtcProgram)prog, name_expression, lowered_name ), GetLoweredName( (hiprtcProgram)prog, name_expression, lowered_name ) );
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
orortcResult OROAPI orortcGetBitcode(orortcProgram prog, char* bitcode)
{
	__ORORTC_FUNC1( GetCUBIN( (nvrtcProgram)prog, bitcode ), 
		GetBitcode( (hiprtcProgram)prog, bitcode ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetBitcodeSize(orortcProgram prog, size_t* bitcodeSizeRet)
{
	__ORORTC_FUNC1( GetCUBINSize( (nvrtcProgram)prog, bitcodeSizeRet ), 
		GetBitcodeSize( (hiprtcProgram)prog, bitcodeSizeRet ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCode(orortcProgram prog, char* code)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		std::ifstream f( getCompiledFilePath() );
		const auto b = f.tellg();
		f.seekg( 0, std::ios::end );
		const auto e = f.tellg();
		const auto s = e - b;
		f.seekg( 0, std::ios::beg );
		f.read( code, s );
		f.close();
		return ORORTC_SUCCESS;
	}
#endif
	__ORORTC_FUNC1( GetPTX( (nvrtcProgram)prog, code ), 
		GetCode( (hiprtcProgram)prog, code ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCodeSize(orortcProgram prog, size_t* codeSizeRet)
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		std::ifstream f( getCompiledFilePath() );
		const auto b = f.tellg();
		f.seekg( 0, std::ios::end );
		const auto e = f.tellg();
		*codeSizeRet = e - b;
		f.close();
		return ORORTC_SUCCESS;
	}
#endif
	__ORORTC_FUNC1( GetPTXSize( (nvrtcProgram)prog, codeSizeRet ), 
		GetCodeSize( (hiprtcProgram)prog, codeSizeRet ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}

orortcResult OROAPI orortcLinkCreate( unsigned int num_options, orortcJIT_option* option_ptr, void** option_vals_pptr, orortcLinkState* link_state_ptr ) 
{ 
	if( s_api & ORO_API_CUDADRIVER ) 
		return cu2orortc( cuLinkCreate( num_options, (CUjit_option*)option_ptr, option_vals_pptr, (CUlinkState*)link_state_ptr ) );
	else
		return hiprtc2oro( hiprtcLinkCreate( num_options, (hiprtcJIT_option*)option_ptr, option_vals_pptr, (hiprtcLinkState*)link_state_ptr ) );

	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkAddFile( orortcLinkState link_state_ptr, orortcJITInputType input_type, const char* file_path, unsigned int num_options, orortcJIT_option* options_ptr, void** option_values ) 
{
	if( s_api & ORO_API_CUDADRIVER )
		return cu2orortc( cuLinkAddFile( (CUlinkState)link_state_ptr, (CUjitInputType)input_type, file_path, num_options, (CUjit_option*)options_ptr, option_values ) );
	else
		return hiprtc2oro( hiprtcLinkAddFile( (hiprtcLinkState)link_state_ptr, (hiprtcJITInputType)input_type, file_path, num_options, (hiprtcJIT_option*)options_ptr, option_values ) );
	return ORORTC_ERROR_INTERNAL_ERROR; 
}
orortcResult OROAPI orortcLinkAddData( orortcLinkState link_state_ptr, orortcJITInputType input_type, void* image, size_t image_size, const char* name, unsigned int num_options, orortcJIT_option* options_ptr, void** option_values ) 
{
	if( s_api & ORO_API_CUDADRIVER )
		return cu2orortc( cuLinkAddData( (CUlinkState)link_state_ptr, (CUjitInputType)input_type, image, image_size, name, num_options, (CUjit_option*)options_ptr ,option_values ) );
	else
		return hiprtc2oro( hiprtcLinkAddData( (hiprtcLinkState)link_state_ptr, (hiprtcJITInputType)input_type, image, image_size, name, num_options, (hiprtcJIT_option*)options_ptr, option_values ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkComplete( orortcLinkState link_state_ptr, void** bin_out, size_t* size_out ) 
{
	if( s_api & ORO_API_CUDADRIVER )
		return cu2orortc( cuLinkComplete( (CUlinkState)link_state_ptr, bin_out, size_out ) );
	else
		return hiprtc2oro( hiprtcLinkComplete( (hiprtcLinkState)link_state_ptr, bin_out, size_out ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkDestroy( orortcLinkState link_state_ptr ) 
{ 
	if( s_api & ORO_API_CUDADRIVER )
		return cu2orortc( cuLinkDestroy( (CUlinkState)link_state_ptr ) );
	else
		return hiprtc2oro( hiprtcLinkDestroy( (hiprtcLinkState)link_state_ptr ) );

	return ORORTC_ERROR_INTERNAL_ERROR; 
}

// Implementation of oroPointerGetAttributes is hacky due to differences between CUDA and HIP
oroError OROAPI oroPointerGetAttributes(oroPointerAttribute* attr, oroDeviceptr dptr)
{
	if (s_api & ORO_API_CUDADRIVER)
	{
		unsigned int data;
		return cu2oro(cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr));
	}
	if (s_api == ORO_API_HIP) 
		return hip2oro(hipPointerGetAttributes((hipPointerAttribute_t*)attr, (void*)dptr));

	return oroErrorUnknown;
}

//-----------------
oroError OROAPI oroStreamCreate( oroStream* stream )
{
#if defined( ORO_INTEL )
	if( s_api & ORO_API_INTEL )
	{
		if( s_iCtxt.m_ctxt == 0 || s_iCtxt.m_device == 0 ) return oroErrorUnknown;

		auto dev = s_iCtxt.m_device;
		uint32_t numQueueGroups = 0;
		auto e = zeDeviceGetCommandQueueGroupProperties( dev, &numQueueGroups, nullptr );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;

		std::vector<ze_command_queue_group_properties_t> queueProperties( numQueueGroups );
		e = zeDeviceGetCommandQueueGroupProperties( dev, &numQueueGroups, queueProperties.data() );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;

		ze_command_queue_desc_t cmdQueueDesc = {};
		for( uint32_t i = 0; i < numQueueGroups; i++ )
		{
			if( queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE )
			{
				cmdQueueDesc.ordinal = i;
			}
		}

		cmdQueueDesc.index = 0;
		cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
		e = zeCommandQueueCreate( s_iCtxt.m_ctxt, dev, &cmdQueueDesc, (ze_command_queue_handle_t*)stream );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		s_iCtxt.m_queue = *(ze_command_queue_handle_t*)stream;

		ze_command_list_handle_t command_list;
		ze_command_list_desc_t cmdListDesc = {};
		cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;
		e = zeCommandListCreate( s_iCtxt.m_ctxt, dev, &cmdListDesc, &command_list );
		s_iCtxt.m_cmdList = command_list;
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;

		return oroSuccess;
	}
#endif
	__ORO_FUNC1(StreamCreate((CUstream*)stream, 0),
		StreamCreate((hipStream_t*)stream));

	return oroErrorUnknown;
}
oroError OROAPI oroStreamSynchronize( oroStream hStream ) 
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		if( hStream == 0 ) hStream = (oroStream)s_iCtxt.m_queue;
		ze_command_queue_handle_t queue = (ze_command_queue_handle_t)hStream;
		auto e = zeCommandListClose( s_iCtxt.m_cmdList );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = zeCommandQueueExecuteCommandLists( queue, 1, &s_iCtxt.m_cmdList, nullptr );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = zeCommandQueueSynchronize(queue, std::numeric_limits<uint64_t>::max() );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = zeCommandListReset( s_iCtxt.m_cmdList );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1( StreamSynchronize( (CUstream)hStream ), StreamSynchronize( (hipStream_t)hStream ) );
	return oroErrorUnknown; 
}
oroError OROAPI oroStreamDestroy( oroStream stream )
{
#if defined( ORO_INTEL )
	if( s_api == ORO_API_INTEL )
	{
		auto e = zeCommandListDestroy( s_iCtxt.m_cmdList );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		e = zeCommandQueueDestroy( (ze_command_queue_handle_t)stream );
		if( e != ZE_RESULT_SUCCESS ) return (oroError)e;
		return oroSuccess;
	}
#endif
	__ORO_FUNC1(StreamDestroy((CUstream)stream), 
		StreamDestroy((hipStream_t)stream ));

	return oroErrorUnknown;
}

//-----------------
oroError OROAPI oroEventCreateWithFlags(oroEvent* phEvent, unsigned int Flags) 
{
	__ORO_FUNC1(EventCreate((CUevent*)phEvent, Flags), 
		EventCreateWithFlags((hipEvent_t*)phEvent, Flags));
	return oroErrorUnknown;
}
oroError OROAPI oroEventRecord(oroEvent hEvent, oroStream hStream ) 
{
	__ORO_FUNC1(EventRecord((CUevent)hEvent, (CUstream)hStream ), 
		EventRecord((hipEvent_t)hEvent, (hipStream_t)hStream));
	return oroErrorUnknown;
}
oroError OROAPI oroEventSynchronize(oroEvent hEvent)
{
	__ORO_FUNC1(EventSynchronize((CUevent)hEvent), 
		EventSynchronize((hipEvent_t)hEvent));
	return oroErrorUnknown;
}
oroError OROAPI oroEventElapsedTime(float* pMilliseconds, oroEvent hStart, oroEvent hEnd)
{
	__ORO_FUNC1(EventElapsedTime(pMilliseconds, (CUevent)hStart, (CUevent)hEnd), 
		EventElapsedTime(pMilliseconds, (hipEvent_t)hStart, (hipEvent_t)hEnd));
	return oroErrorUnknown;
}
oroError OROAPI oroEventDestroy(oroEvent hEvent) 
{
	__ORO_FUNC1(EventDestroy((CUevent)hEvent), 
		EventDestroy((hipEvent_t)hEvent));
	return oroErrorUnknown;
}
