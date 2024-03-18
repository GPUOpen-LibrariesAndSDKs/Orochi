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
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <mutex>


// this namespace will be for encapsulating all the files related to CUDA
// we need to make that into a workspace espacially because of the files from HIP SDK:  nvidia_hip_runtime_api.h  /  nvidia_hiprtc.h.  that creates new definitions of hip**** functions
namespace CU4ORO 
{
#include <contrib/cuew/include/cuew.h>
#ifdef OROCHI_CUEW_DEFINED
#include "nvidia_hip_runtime_api_oro.h"
#include "nvidia_hiprtc_oro.h"
#endif
}





std::unordered_map<void*, oroCtx> s_oroCtxs;
static std::mutex mtx;
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
	oroGetDeviceCount( &n[1], ORO_API_CUDADRIVER );

	oroApi api = (deviceId < n[0]) ? (ORO_API_HIP) : (ORO_API_CUDADRIVER);
	if( api & ORO_API_CUDADRIVER )
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
		#ifdef OROCHI_CUEW_DEFINED
		uint32_t flag = 0;
		if( api & ORO_API_CUDADRIVER )
		{
			flag |= CU4ORO::CUEW_INIT_CUDA;
		}
		if( api & ORO_API_CUDARTC )
		{
			flag |= CU4ORO::CUEW_INIT_NVRTC;
		}
		
		int resultDriver, resultRtc;
		CU4ORO::cuewInit( &resultDriver, &resultRtc, flag );

		if( resultDriver == CU4ORO::CUEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_CUDADRIVER;
		}
		if( resultRtc == CU4ORO::CUEW_SUCCESS )
		{
			s_loadedApis |= ORO_API_CUDARTC;
		}
		#endif
	}
	if( api & ORO_API_HIP )
	{
		uint32_t flag = 0;
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


inline oroError hip2oro( hipError_t a ) { return a; }
inline orortcResult hip2oro( hiprtcResult a ) { return a; }
inline const char * hip2oro( const char * a ) { return a; }


#ifdef OROCHI_CUEW_DEFINED

inline oroError cu2oro( CU4ORO::hipError_t a ) { return (hipError_t)a; } //  CU4ORO::hipError_t = hipError_t, so we can cast it safely
inline orortcResult cu2oro( CU4ORO::hiprtcResult a ) { return (hiprtcResult)a; } //  CU4ORO::hiprtcResult = hiprtcResult, so we can cast it safely
inline oroError cu2oro( CU4ORO::CUresult a ) { return cu2oro(CU4ORO::hipCUResultTohipError(a)); }
inline orortcResult cu2oro( CU4ORO::nvrtcResult a ) { return cu2oro(CU4ORO::nvrtcResultTohiprtcResult(a)); }
inline const char * cu2oro( const char * a ) { return a; }
inline oroError cu2oro( CU4ORO::cudaError_t a ) { return cu2oro(CU4ORO::hipCUDAErrorTohipError(a)); }

// not a natural cast, but may be needed sometimes
CU4ORO::nvrtcResult cu2nvrtc(CU4ORO::CUresult a) 
{
	switch (a) {
	case CU4ORO::CUDA_SUCCESS:
		return CU4ORO::NVRTC_SUCCESS;
	case CU4ORO::CUDA_ERROR_OUT_OF_MEMORY:
		return CU4ORO::NVRTC_ERROR_OUT_OF_MEMORY;
	case CU4ORO::CUDA_ERROR_INVALID_VALUE:
		return CU4ORO::NVRTC_ERROR_INVALID_INPUT;
	default:
		return CU4ORO::NVRTC_ERROR_INTERNAL_ERROR;
	}
} 
  
inline CU4ORO::CUcontext* oroCtx2cu( oroCtx* a )
{
	ioroCtx_t* b = *a;
	return (CU4ORO::CUcontext*)&b->m_ptr;
}

#endif // OROCHI_CUEW_DEFINED




inline hipCtx_t* oroCtx2hip( oroCtx* a )
{
	ioroCtx_t* b = *a;
	return (hipCtx_t*)&b->m_ptr;
}






#ifdef OROCHI_CUEW_DEFINED
#define __ORO_FUNCX( API, cuname, hipname ) if( API & ORO_API_CUDADRIVER ) return cu2oro( cuname ); if( API == ORO_API_HIP ) return hip2oro( hipname );
#define __ORO_FUNC(cuname,hipname) if( s_api & ORO_API_CUDADRIVER ) return cu2oro(cuname); if( s_api == ORO_API_HIP ) return hip2oro(hipname);
#else
#define __ORO_FUNCX( API, cuname, hipname )  if( API == ORO_API_HIP ) return hip2oro( hip##hipname );
#define __ORO_FUNC(cuname,hipname)  if( s_api == ORO_API_HIP ) return hipname;
#endif

#define __ORO_FORCE_CAST(type,var)     *((type*)(&var))




oroError OROAPI oroGetErrorString( oroError error, const char** pStr )
{
	if( s_api & ORO_API_CUDADRIVER ) 
	{
		#ifdef OROCHI_CUEW_DEFINED
		return cu2oro(CU4ORO::cuGetErrorString( (CU4ORO::CUresult)error, pStr ));
		#endif
	}
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
		#ifdef OROCHI_CUEW_DEFINED
		e1 = cu2oro( CU4ORO::cuInit( Flags ) );
		#endif
	}
	return ( e0 == 0 || e1 == 0 ) ? oroSuccess : oroErrorUnknown;
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
		
		#ifdef OROCHI_CUEW_DEFINED
		int c = 0;
		e = cu2oro(CU4ORO::cuDeviceGetCount(&c));
		if( e == 0 )
			*count += c;
		#endif
	}
	return oroSuccess;
}



oroError OROAPI oroGetDeviceProperties(oroDeviceProp_t* props, oroDevice dev)
{
	ioroDevice d( dev );
	int deviceId = d.getDevice();
	oroApi api = d.getApi();
	*props = {};
	if( api == ORO_API_HIP )
		return hip2oro(hipGetDeviceProperties(props, deviceId));
	if( api & ORO_API_CUDADRIVER )
	{
		#ifdef OROCHI_CUEW_DEFINED
		return  (oroError_t)( CU4ORO::hipGetDeviceProperties( (CU4ORO::hipDeviceProp_t*) props, deviceId) );
		#endif
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
	if (api & ORO_API_CUDADRIVER)
	{
		#ifdef OROCHI_CUEW_DEFINED
		int t;
		auto e = CU4ORO::cuDeviceGet(&t, ordinal);
		d.setApi(api);
		d.setDevice(t);
		*(ioroDevice*)device = d;
		return cu2oro(e);
		#endif
	}
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceGetName(char* name, int len, oroDevice dev)
{
	ioroDevice d( dev );
	__ORO_FUNCX( d.getApi(), 
		CU4ORO::cuDeviceGetName(name, len, d.getDevice() ),
		hipDeviceGetName(name, len, d.getDevice() ) 
		);
	return oroErrorUnknown;
}

oroError OROAPI oroDeviceGetAttribute(int* pi, oroDeviceAttribute_t attrib, oroDevice dev)
{
	ioroDevice d( dev );
	__ORO_FUNCX( d.getApi(), CU4ORO::cuDeviceGetAttribute( pi, (CU4ORO::CUdevice_attribute)attrib, d.getDevice() ), hipDeviceGetAttribute( pi, (hipDeviceAttribute_t)attrib, d.getDevice() ) );
	return oroErrorUnknown;
}

oroError OROAPI oroCtxCreate(oroCtx* pctx, unsigned int flags, oroDevice dev)
{
	ioroDevice d( dev );
	ioroCtx_t* ctxt = new ioroCtx_t;
	ctxt->setApi( d.getApi() );
	(*pctx) = ctxt;
	s_api = ctxt->getApi();
	if( s_api & ORO_API_CUDADRIVER ) 
	{
		#ifdef OROCHI_CUEW_DEFINED
		CU4ORO::CUresult e = CU4ORO::cuCtxCreate( oroCtx2cu( pctx ), flags, d.getDevice() );
		if ( e != CU4ORO::CUDA_SUCCESS )
			return cu2oro(e);
		#endif
	}
	if( s_api == ORO_API_HIP ) 
	{
		hipError_t e = hipCtxCreate( oroCtx2hip( pctx ), flags, d.getDevice() );
		if ( e != hipSuccess )
			return hip2oro(e);
	}
	std::lock_guard<std::mutex> lock( mtx );
	s_oroCtxs[ctxt->m_ptr] = ctxt;
	return oroSuccess;
}

oroError OROAPI oroCtxDestroy(oroCtx ctx)
{
	std::lock_guard<std::mutex> lock( mtx );
	s_oroCtxs.erase( ctx->m_ptr );

	int e = 0;
	if( s_api & ORO_API_CUDADRIVER )
	{
		#ifdef OROCHI_CUEW_DEFINED
		e = CU4ORO::cuCtxDestroy( *oroCtx2cu( &ctx ) );
		#endif
	}
	if( s_api == ORO_API_HIP ) e = hipCtxDestroy( *oroCtx2hip( &ctx ) );

	if( e )
		return oroErrorUnknown;
	ioroCtx_t* c = (ioroCtx_t*)ctx;
	delete c;
	return oroSuccess;
}

oroError OROAPI oroCtxSetCurrent(oroCtx ctx)
{
	s_api = ctx->getApi();
	__ORO_FUNC(
		CU4ORO::hipCtxSetCurrent( *oroCtx2cu(&ctx) ),
				hipCtxSetCurrent( *oroCtx2hip(&ctx) )  );
	return oroErrorUnknown;
}

oroError OROAPI oroCtxGetCurrent(oroCtx* pctx)
{
	ioroCtx_t* ctxt = new ioroCtx_t;

	if( s_api & ORO_API_CUDADRIVER ) 
	{
		#ifdef OROCHI_CUEW_DEFINED
		CU4ORO::CUresult e = CU4ORO::cuCtxGetCurrent( oroCtx2cu( &ctxt ) );
		if ( e != CU4ORO::CUDA_SUCCESS )
			return cu2oro(e);
		#endif
	}
	if( s_api == ORO_API_HIP ) 
	{
		hipError_t e = hipCtxGetCurrent( oroCtx2hip( &ctxt ) );
		if ( e != hipSuccess )
			return hip2oro(e);
	}
	( *pctx ) = s_oroCtxs[ctxt->m_ptr];
	delete ctxt;
	return oroSuccess;
}

oroError OROAPI oroCtxGetApiVersion(oroCtx ctx, int* version)
{
	__ORO_FUNC(
	CU4ORO::hipCtxGetApiVersion(*oroCtx2cu(&ctx),  version ),
			hipCtxGetApiVersion(*oroCtx2hip(&ctx), version )  );
	return oroErrorUnknown;
}



// function can't be automatically generated because returning a structure.
oroChannelFormatDesc OROAPI oroCreateChannelDesc(int x, int y, int z, int w,  oroChannelFormatKind f)
{
	 if( s_api & ORO_API_CUDADRIVER )
	 {
		#ifdef OROCHI_CUEW_DEFINED
		CU4ORO::hipChannelFormatDesc ret = CU4ORO::hipCreateChannelDesc(__ORO_FORCE_CAST(int,x), __ORO_FORCE_CAST(int,y), __ORO_FORCE_CAST(int,z), __ORO_FORCE_CAST(int,w), __ORO_FORCE_CAST(CU4ORO::hipChannelFormatKind,f));
		return __ORO_FORCE_CAST(oroChannelFormatDesc, ret);
		#endif
	 }
	 if( s_api == ORO_API_HIP ) 
		 return hipCreateChannelDesc(x, y, z, w, f);

	return oroChannelFormatDesc();
}

orortcResult OROAPI orortcGetBitcode(orortcProgram prog, char* bitcode)
{
	__ORO_FUNC( CU4ORO::nvrtcGetCUBIN( (CU4ORO::nvrtcProgram)prog, bitcode ), 
				  hiprtcGetBitcode( prog, bitcode ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}

orortcResult OROAPI orortcGetBitcodeSize(orortcProgram prog, size_t* bitcodeSizeRet)
{
	__ORO_FUNC(
		CU4ORO::nvrtcGetCUBINSize( (CU4ORO::nvrtcProgram)prog, bitcodeSizeRet ), 
		hiprtcGetBitcodeSize( (hiprtcProgram)prog, bitcodeSizeRet ) );
	return ORORTC_ERROR_INTERNAL_ERROR;
}





#pragma region OROCHI_SUMMONER_REGION_orochi_cpp

/////
///// THIS REGION HAS BEEN AUTOMATICALLY GENERATED BY OROCHI SUMMONER.
///// Manual modification of this region is not recommended.
/////

oroError_t OROAPI oroChooseDeviceR0600(int * device, const oroDeviceProp_tR0600 * prop)
{
	__ORO_FUNC(
		CU4ORO::hipChooseDeviceR0600(__ORO_FORCE_CAST(int *,device), __ORO_FORCE_CAST(const CU4ORO::hipDeviceProp_t *,prop)),
		hipChooseDeviceR0600(device, prop)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCreateSurfaceObject(oroSurfaceObject_t * pSurfObject, const oroResourceDesc * pResDesc)
{
	__ORO_FUNC(
		CU4ORO::hipCreateSurfaceObject(__ORO_FORCE_CAST(CU4ORO::hipSurfaceObject_t *,pSurfObject), __ORO_FORCE_CAST(const CU4ORO::hipResourceDesc *,pResDesc)),
		hipCreateSurfaceObject(pSurfObject, pResDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCreateTextureObject(oroTextureObject_t * pTexObject, const oroResourceDesc * pResDesc, const oroTextureDesc * pTexDesc, const  oroResourceViewDesc * pResViewDesc)
{
	__ORO_FUNC(
		CU4ORO::hipCreateTextureObject(__ORO_FORCE_CAST(CU4ORO::hipTextureObject_t *,pTexObject), __ORO_FORCE_CAST(const CU4ORO::hipResourceDesc *,pResDesc), __ORO_FORCE_CAST(const CU4ORO::hipTextureDesc *,pTexDesc), __ORO_FORCE_CAST(const CU4ORO::hipResourceViewDesc *,pResViewDesc)),
		hipCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxDisablePeerAccess(oroCtx_t peerCtx)
{
	__ORO_FUNC(
		CU4ORO::hipCtxDisablePeerAccess(__ORO_FORCE_CAST(CU4ORO::hipCtx_t,peerCtx)),
		hipCtxDisablePeerAccess(peerCtx)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxEnablePeerAccess(oroCtx_t peerCtx, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipCtxEnablePeerAccess(__ORO_FORCE_CAST(CU4ORO::hipCtx_t,peerCtx), __ORO_FORCE_CAST(unsigned int,flags)),
		hipCtxEnablePeerAccess(peerCtx, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxGetCacheConfig(oroFuncCache_t * cacheConfig)
{
	__ORO_FUNC(
		CU4ORO::hipCtxGetCacheConfig(__ORO_FORCE_CAST(CU4ORO::hipFuncCache *,cacheConfig)),
		hipCtxGetCacheConfig(cacheConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxGetDevice(oroDevice_t * device)
{
	__ORO_FUNC(
		CU4ORO::hipCtxGetDevice(__ORO_FORCE_CAST(CU4ORO::hipDevice_t *,device)),
		hipCtxGetDevice(device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxGetFlags(unsigned int * flags)
{
	__ORO_FUNC(
		CU4ORO::hipCtxGetFlags(__ORO_FORCE_CAST(unsigned int *,flags)),
		hipCtxGetFlags(flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxGetSharedMemConfig(oroSharedMemConfig * pConfig)
{
	__ORO_FUNC(
		CU4ORO::hipCtxGetSharedMemConfig(__ORO_FORCE_CAST(CU4ORO::hipSharedMemConfig *,pConfig)),
		hipCtxGetSharedMemConfig(pConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxPopCurrent(oroCtx_t * ctx)
{
	__ORO_FUNC(
		CU4ORO::hipCtxPopCurrent(__ORO_FORCE_CAST(CU4ORO::hipCtx_t *,ctx)),
		hipCtxPopCurrent(ctx)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxPushCurrent(oroCtx_t ctx)
{
	__ORO_FUNC(
		CU4ORO::hipCtxPushCurrent(__ORO_FORCE_CAST(CU4ORO::hipCtx_t,ctx)),
		hipCtxPushCurrent(ctx)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxSetCacheConfig(oroFuncCache_t cacheConfig)
{
	__ORO_FUNC(
		CU4ORO::hipCtxSetCacheConfig(__ORO_FORCE_CAST(CU4ORO::hipFuncCache,cacheConfig)),
		hipCtxSetCacheConfig(cacheConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxSetSharedMemConfig(oroSharedMemConfig config)
{
	__ORO_FUNC(
		CU4ORO::hipCtxSetSharedMemConfig(__ORO_FORCE_CAST(CU4ORO::hipSharedMemConfig,config)),
		hipCtxSetSharedMemConfig(config)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroCtxSynchronize()
{
	__ORO_FUNC(
		CU4ORO::hipCtxSynchronize(),
		hipCtxSynchronize()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDestroyExternalMemory(oroExternalMemory_t extMem)
{
	__ORO_FUNC(
		CU4ORO::hipDestroyExternalMemory(__ORO_FORCE_CAST(CU4ORO::hipExternalMemory_t,extMem)),
		hipDestroyExternalMemory(extMem)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDestroyExternalSemaphore(oroExternalSemaphore_t extSem)
{
	__ORO_FUNC(
		CU4ORO::hipDestroyExternalSemaphore(__ORO_FORCE_CAST(CU4ORO::hipExternalSemaphore_t,extSem)),
		hipDestroyExternalSemaphore(extSem)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDestroySurfaceObject(oroSurfaceObject_t surfaceObject)
{
	__ORO_FUNC(
		CU4ORO::hipDestroySurfaceObject(__ORO_FORCE_CAST(CU4ORO::hipSurfaceObject_t,surfaceObject)),
		hipDestroySurfaceObject(surfaceObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDestroyTextureObject(oroTextureObject_t textureObject)
{
	__ORO_FUNC(
		CU4ORO::hipDestroyTextureObject(__ORO_FORCE_CAST(CU4ORO::hipTextureObject_t,textureObject)),
		hipDestroyTextureObject(textureObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceCanAccessPeer(int * canAccessPeer, int deviceId, int peerDeviceId)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceCanAccessPeer(__ORO_FORCE_CAST(int *,canAccessPeer), __ORO_FORCE_CAST(int,deviceId), __ORO_FORCE_CAST(int,peerDeviceId)),
		hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceComputeCapability(int * major, int * minor, oroDevice_t device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceComputeCapability(__ORO_FORCE_CAST(int *,major), __ORO_FORCE_CAST(int *,minor), __ORO_FORCE_CAST(CU4ORO::hipDevice_t,device)),
		hipDeviceComputeCapability(major, minor, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceDisablePeerAccess(int peerDeviceId)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceDisablePeerAccess(__ORO_FORCE_CAST(int,peerDeviceId)),
		hipDeviceDisablePeerAccess(peerDeviceId)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceEnablePeerAccess(__ORO_FORCE_CAST(int,peerDeviceId), __ORO_FORCE_CAST(unsigned int,flags)),
		hipDeviceEnablePeerAccess(peerDeviceId, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetByPCIBusId(int * device, const char * pciBusId)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetByPCIBusId(__ORO_FORCE_CAST(int *,device), __ORO_FORCE_CAST(const char *,pciBusId)),
		hipDeviceGetByPCIBusId(device, pciBusId)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetCacheConfig(oroFuncCache_t * cacheConfig)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetCacheConfig(__ORO_FORCE_CAST(CU4ORO::hipFuncCache_t *,cacheConfig)),
		hipDeviceGetCacheConfig(cacheConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetDefaultMemPool(oroMemPool_t * mem_pool, int device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetDefaultMemPool(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t *,mem_pool), __ORO_FORCE_CAST(int,device)),
		hipDeviceGetDefaultMemPool(mem_pool, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetLimit(size_t * pValue,  oroLimit_t limit)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetLimit(__ORO_FORCE_CAST(size_t *,pValue), __ORO_FORCE_CAST(CU4ORO::hipLimit_t,limit)),
		hipDeviceGetLimit(pValue, limit)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetMemPool(oroMemPool_t * mem_pool, int device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetMemPool(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t *,mem_pool), __ORO_FORCE_CAST(int,device)),
		hipDeviceGetMemPool(mem_pool, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetP2PAttribute(int * value, oroDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetP2PAttribute(__ORO_FORCE_CAST(int *,value), __ORO_FORCE_CAST(CU4ORO::hipDeviceP2PAttr,attr), __ORO_FORCE_CAST(int,srcDevice), __ORO_FORCE_CAST(int,dstDevice)),
		hipDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetPCIBusId(char * pciBusId, int len, int device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetPCIBusId(__ORO_FORCE_CAST(char *,pciBusId), __ORO_FORCE_CAST(int,len), __ORO_FORCE_CAST(CU4ORO::hipDevice_t,device)),
		hipDeviceGetPCIBusId(pciBusId, len, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetSharedMemConfig(oroSharedMemConfig * pConfig)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetSharedMemConfig(__ORO_FORCE_CAST(CU4ORO::hipSharedMemConfig *,pConfig)),
		hipDeviceGetSharedMemConfig(pConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetStreamPriorityRange(__ORO_FORCE_CAST(int *,leastPriority), __ORO_FORCE_CAST(int *,greatestPriority)),
		hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetUuid(oroUUID * uuid, oroDevice_t device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceGetUuid(__ORO_FORCE_CAST(CU4ORO::hipUUID *,uuid), __ORO_FORCE_CAST(CU4ORO::hipDevice_t,device)),
		hipDeviceGetUuid(uuid, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDevicePrimaryCtxGetState(oroDevice_t dev, unsigned int * flags, int * active)
{
	__ORO_FUNC(
		CU4ORO::hipDevicePrimaryCtxGetState(__ORO_FORCE_CAST(CU4ORO::hipDevice_t,dev), __ORO_FORCE_CAST(unsigned int *,flags), __ORO_FORCE_CAST(int *,active)),
		hipDevicePrimaryCtxGetState(dev, flags, active)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDevicePrimaryCtxRelease(oroDevice_t dev)
{
	__ORO_FUNC(
		CU4ORO::hipDevicePrimaryCtxRelease(__ORO_FORCE_CAST(CU4ORO::hipDevice_t,dev)),
		hipDevicePrimaryCtxRelease(dev)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDevicePrimaryCtxReset(oroDevice_t dev)
{
	__ORO_FUNC(
		CU4ORO::hipDevicePrimaryCtxReset(__ORO_FORCE_CAST(CU4ORO::hipDevice_t,dev)),
		hipDevicePrimaryCtxReset(dev)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDevicePrimaryCtxRetain(oroCtx_t * pctx, oroDevice_t dev)
{
	__ORO_FUNC(
		CU4ORO::hipDevicePrimaryCtxRetain(__ORO_FORCE_CAST(CU4ORO::hipCtx_t *,pctx), __ORO_FORCE_CAST(CU4ORO::hipDevice_t,dev)),
		hipDevicePrimaryCtxRetain(pctx, dev)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDevicePrimaryCtxSetFlags(oroDevice_t dev, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipDevicePrimaryCtxSetFlags(__ORO_FORCE_CAST(CU4ORO::hipDevice_t,dev), __ORO_FORCE_CAST(unsigned int,flags)),
		hipDevicePrimaryCtxSetFlags(dev, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceReset()
{
	__ORO_FUNC(
		CU4ORO::hipDeviceReset(),
		hipDeviceReset()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSetCacheConfig(oroFuncCache_t cacheConfig)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceSetCacheConfig(__ORO_FORCE_CAST(CU4ORO::hipFuncCache_t,cacheConfig)),
		hipDeviceSetCacheConfig(cacheConfig)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSetLimit( oroLimit_t limit, size_t value)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceSetLimit(__ORO_FORCE_CAST(CU4ORO::hipLimit_t,limit), __ORO_FORCE_CAST(size_t,value)),
		hipDeviceSetLimit(limit, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSetMemPool(int device, oroMemPool_t mem_pool)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceSetMemPool(__ORO_FORCE_CAST(int,device), __ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool)),
		hipDeviceSetMemPool(device, mem_pool)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSetSharedMemConfig(oroSharedMemConfig config)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceSetSharedMemConfig(__ORO_FORCE_CAST(CU4ORO::hipSharedMemConfig,config)),
		hipDeviceSetSharedMemConfig(config)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSynchronize()
{
	__ORO_FUNC(
		CU4ORO::hipDeviceSynchronize(),
		hipDeviceSynchronize()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceTotalMem(size_t * bytes, oroDevice_t device)
{
	__ORO_FUNC(
		CU4ORO::hipDeviceTotalMem(__ORO_FORCE_CAST(size_t *,bytes), __ORO_FORCE_CAST(CU4ORO::hipDevice_t,device)),
		hipDeviceTotalMem(bytes, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDriverGetVersion(int * driverVersion)
{
	__ORO_FUNC(
		CU4ORO::hipDriverGetVersion(__ORO_FORCE_CAST(int *,driverVersion)),
		hipDriverGetVersion(driverVersion)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvGetErrorName(oroError_t hipError, const char ** errorString)
{
	__ORO_FUNC(
		CU4ORO::hipDrvGetErrorName(__ORO_FORCE_CAST(CU4ORO::hipError_t,hipError), __ORO_FORCE_CAST(const char **,errorString)),
		hipDrvGetErrorName(hipError, errorString)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvGetErrorString(oroError_t hipError, const char ** errorString)
{
	__ORO_FUNC(
		CU4ORO::hipDrvGetErrorString(__ORO_FORCE_CAST(CU4ORO::hipError_t,hipError), __ORO_FORCE_CAST(const char **,errorString)),
		hipDrvGetErrorString(hipError, errorString)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvMemcpy3D(const ORO_MEMCPY3D * pCopy)
{
	__ORO_FUNC(
		CU4ORO::hipDrvMemcpy3D(__ORO_FORCE_CAST(const CU4ORO::HIP_MEMCPY3D *,pCopy)),
		hipDrvMemcpy3D(pCopy)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvMemcpy3DAsync(const ORO_MEMCPY3D * pCopy, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipDrvMemcpy3DAsync(__ORO_FORCE_CAST(const CU4ORO::HIP_MEMCPY3D *,pCopy), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipDrvMemcpy3DAsync(pCopy, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvPointerGetAttributes(unsigned int numAttributes, oroPointer_attribute * attributes, void ** data, oroDeviceptr_t ptr)
{
	__ORO_FUNC(
		CU4ORO::hipDrvPointerGetAttributes(__ORO_FORCE_CAST(unsigned int,numAttributes), __ORO_FORCE_CAST(CU4ORO::CUpointer_attribute *,attributes), __ORO_FORCE_CAST(void **,data), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr)),
		hipDrvPointerGetAttributes(numAttributes, attributes, data, ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventCreate(oroEvent_t * event)
{
	__ORO_FUNC(
		CU4ORO::hipEventCreate(__ORO_FORCE_CAST(CU4ORO::hipEvent_t *,event)),
		hipEventCreate(event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventCreateWithFlags(oroEvent_t * event, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipEventCreateWithFlags(__ORO_FORCE_CAST(CU4ORO::hipEvent_t *,event), __ORO_FORCE_CAST(unsigned int,flags)),
		hipEventCreateWithFlags(event, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventDestroy(oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::hipEventDestroy(__ORO_FORCE_CAST(CU4ORO::hipEvent_t,event)),
		hipEventDestroy(event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventElapsedTime(float * ms, oroEvent_t start, oroEvent_t stop)
{
	__ORO_FUNC(
		CU4ORO::hipEventElapsedTime(__ORO_FORCE_CAST(float *,ms), __ORO_FORCE_CAST(CU4ORO::hipEvent_t,start), __ORO_FORCE_CAST(CU4ORO::hipEvent_t,stop)),
		hipEventElapsedTime(ms, start, stop)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventQuery(oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::hipEventQuery(__ORO_FORCE_CAST(CU4ORO::hipEvent_t,event)),
		hipEventQuery(event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventRecord(oroEvent_t event, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipEventRecord(__ORO_FORCE_CAST(CU4ORO::hipEvent_t,event), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipEventRecord(event, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroEventSynchronize(oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::hipEventSynchronize(__ORO_FORCE_CAST(CU4ORO::hipEvent_t,event)),
		hipEventSynchronize(event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroExternalMemoryGetMappedBuffer(void ** devPtr, oroExternalMemory_t extMem, const oroExternalMemoryBufferDesc * bufferDesc)
{
	__ORO_FUNC(
		CU4ORO::hipExternalMemoryGetMappedBuffer(__ORO_FORCE_CAST(void **,devPtr), __ORO_FORCE_CAST(CU4ORO::hipExternalMemory_t,extMem), __ORO_FORCE_CAST(const CU4ORO::hipExternalMemoryBufferDesc *,bufferDesc)),
		hipExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroExternalMemoryGetMappedMipmappedArray(oroMipmappedArray_t * mipmap, oroExternalMemory_t extMem, const oroExternalMemoryMipmappedArrayDesc * mipmapDesc)
{
	__ORO_FUNC(
		CU4ORO::hipExternalMemoryGetMappedMipmappedArray(__ORO_FORCE_CAST(CU4ORO::hipMipmappedArray_t *,mipmap), __ORO_FORCE_CAST(CU4ORO::hipExternalMemory_t,extMem), __ORO_FORCE_CAST(const CU4ORO::hipExternalMemoryMipmappedArrayDesc *,mipmapDesc)),
		hipExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFree(void * ptr)
{
	__ORO_FUNC(
		CU4ORO::hipFree(__ORO_FORCE_CAST(void *,ptr)),
		hipFree(ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFreeArray(oroArray_t array)
{
	__ORO_FUNC(
		CU4ORO::hipFreeArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t,array)),
		hipFreeArray(array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFreeAsync(void * dev_ptr, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipFreeAsync(__ORO_FORCE_CAST(void *,dev_ptr), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipFreeAsync(dev_ptr, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFreeHost(void * ptr)
{
	__ORO_FUNC(
		CU4ORO::hipFreeHost(__ORO_FORCE_CAST(void *,ptr)),
		hipFreeHost(ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFreeMipmappedArray(oroMipmappedArray_t mipmappedArray)
{
	__ORO_FUNC(
		CU4ORO::hipFreeMipmappedArray(__ORO_FORCE_CAST(CU4ORO::hipMipmappedArray_t,mipmappedArray)),
		hipFreeMipmappedArray(mipmappedArray)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFuncGetAttribute(int * value, oroFunction_attribute attrib, oroFunction_t hfunc)
{
	__ORO_FUNC(
		CU4ORO::hipFuncGetAttribute(__ORO_FORCE_CAST(int *,value), __ORO_FORCE_CAST(CU4ORO::CUfunction_attribute,attrib), __ORO_FORCE_CAST(CU4ORO::hipFunction_t,hfunc)),
		hipFuncGetAttribute(value, attrib, hfunc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFuncGetAttributes( oroFuncAttributes * attr, const void * func)
{
	__ORO_FUNC(
		CU4ORO::hipFuncGetAttributes(__ORO_FORCE_CAST(CU4ORO::hipFuncAttributes *,attr), __ORO_FORCE_CAST(const void *,func)),
		hipFuncGetAttributes(attr, func)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFuncSetAttribute(const void * func, oroFuncAttribute attr, int value)
{
	__ORO_FUNC(
		CU4ORO::hipFuncSetAttribute(__ORO_FORCE_CAST(const void *,func), __ORO_FORCE_CAST(CU4ORO::hipFuncAttribute,attr), __ORO_FORCE_CAST(int,value)),
		hipFuncSetAttribute(func, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFuncSetCacheConfig(const void * func, oroFuncCache_t config)
{
	__ORO_FUNC(
		CU4ORO::hipFuncSetCacheConfig(__ORO_FORCE_CAST(const void *,func), __ORO_FORCE_CAST(CU4ORO::hipFuncCache_t,config)),
		hipFuncSetCacheConfig(func, config)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroFuncSetSharedMemConfig(const void * func, oroSharedMemConfig config)
{
	__ORO_FUNC(
		CU4ORO::hipFuncSetSharedMemConfig(__ORO_FORCE_CAST(const void *,func), __ORO_FORCE_CAST(CU4ORO::hipSharedMemConfig,config)),
		hipFuncSetSharedMemConfig(func, config)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetChannelDesc(oroChannelFormatDesc * desc, oroArray_const_t array)
{
	__ORO_FUNC(
		CU4ORO::hipGetChannelDesc(__ORO_FORCE_CAST(CU4ORO::hipChannelFormatDesc *,desc), __ORO_FORCE_CAST(CU4ORO::hipArray_const_t,array)),
		hipGetChannelDesc(desc, array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetDevice(int * deviceId)
{
	__ORO_FUNC(
		CU4ORO::hipGetDevice(__ORO_FORCE_CAST(int *,deviceId)),
		hipGetDevice(deviceId)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetDeviceFlags(unsigned int * flags)
{
	__ORO_FUNC(
		CU4ORO::hipGetDeviceFlags(__ORO_FORCE_CAST(unsigned int *,flags)),
		hipGetDeviceFlags(flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetDevicePropertiesR0600(oroDeviceProp_tR0600 * prop, int deviceId)
{
	__ORO_FUNC(
		CU4ORO::hipGetDevicePropertiesR0600(__ORO_FORCE_CAST(CU4ORO::hipDeviceProp_t *,prop), __ORO_FORCE_CAST(int,deviceId)),
		hipGetDevicePropertiesR0600(prop, deviceId)     );
	return oroErrorUnknown;
}
const char * OROAPI oroGetErrorName(oroError_t hip_error)
{
	__ORO_FUNC(
		CU4ORO::hipGetErrorName(__ORO_FORCE_CAST(CU4ORO::hipError_t,hip_error)),
		hipGetErrorName(hip_error)     );
	return nullptr;
}
oroError_t OROAPI oroGetLastError()
{
	__ORO_FUNC(
		CU4ORO::hipGetLastError(),
		hipGetLastError()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetMipmappedArrayLevel(oroArray_t * levelArray, oroMipmappedArray_const_t mipmappedArray, unsigned int level)
{
	__ORO_FUNC(
		CU4ORO::hipGetMipmappedArrayLevel(__ORO_FORCE_CAST(CU4ORO::hipArray_t *,levelArray), __ORO_FORCE_CAST(CU4ORO::hipMipmappedArray_t,mipmappedArray), __ORO_FORCE_CAST(unsigned int,level)),
		hipGetMipmappedArrayLevel(levelArray, mipmappedArray, level)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetSymbolAddress(void ** devPtr, const void * symbol)
{
	__ORO_FUNC(
		CU4ORO::hipGetSymbolAddress(__ORO_FORCE_CAST(void **,devPtr), __ORO_FORCE_CAST(const void *,symbol)),
		hipGetSymbolAddress(devPtr, symbol)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetSymbolSize(size_t * size, const void * symbol)
{
	__ORO_FUNC(
		CU4ORO::hipGetSymbolSize(__ORO_FORCE_CAST(size_t *,size), __ORO_FORCE_CAST(const void *,symbol)),
		hipGetSymbolSize(size, symbol)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetTextureObjectResourceDesc(oroResourceDesc * pResDesc, oroTextureObject_t textureObject)
{
	__ORO_FUNC(
		CU4ORO::hipGetTextureObjectResourceDesc(__ORO_FORCE_CAST(CU4ORO::hipResourceDesc *,pResDesc), __ORO_FORCE_CAST(CU4ORO::hipTextureObject_t,textureObject)),
		hipGetTextureObjectResourceDesc(pResDesc, textureObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphicsMapResources(int count, oroGraphicsResource_t * resources, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipGraphicsMapResources(__ORO_FORCE_CAST(int,count), __ORO_FORCE_CAST(CU4ORO::hipGraphicsResource_t *,resources), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipGraphicsMapResources(count, resources, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, oroGraphicsResource_t resource)
{
	__ORO_FUNC(
		CU4ORO::hipGraphicsResourceGetMappedPointer(__ORO_FORCE_CAST(void **,devPtr), __ORO_FORCE_CAST(size_t *,size), __ORO_FORCE_CAST(CU4ORO::hipGraphicsResource_t,resource)),
		hipGraphicsResourceGetMappedPointer(devPtr, size, resource)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphicsSubResourceGetMappedArray(oroArray_t * array, oroGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
	__ORO_FUNC(
		CU4ORO::hipGraphicsSubResourceGetMappedArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t *,array), __ORO_FORCE_CAST(CU4ORO::hipGraphicsResource_t,resource), __ORO_FORCE_CAST(unsigned int,arrayIndex), __ORO_FORCE_CAST(unsigned int,mipLevel)),
		hipGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphicsUnmapResources(int count, oroGraphicsResource_t * resources, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipGraphicsUnmapResources(__ORO_FORCE_CAST(int,count), __ORO_FORCE_CAST(CU4ORO::hipGraphicsResource_t *,resources), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipGraphicsUnmapResources(count, resources, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphicsUnregisterResource(oroGraphicsResource_t resource)
{
	__ORO_FUNC(
		CU4ORO::hipGraphicsUnregisterResource(__ORO_FORCE_CAST(CU4ORO::hipGraphicsResource_t,resource)),
		hipGraphicsUnregisterResource(resource)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostAlloc(void ** ptr, size_t size, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipHostAlloc(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(unsigned int,flags)),
		hipHostAlloc(ptr, size, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostFree(void * ptr)
{
	__ORO_FUNC(
		CU4ORO::hipHostFree(__ORO_FORCE_CAST(void *,ptr)),
		hipHostFree(ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostGetDevicePointer(void ** devPtr, void * hstPtr, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipHostGetDevicePointer(__ORO_FORCE_CAST(void **,devPtr), __ORO_FORCE_CAST(void *,hstPtr), __ORO_FORCE_CAST(unsigned int,flags)),
		hipHostGetDevicePointer(devPtr, hstPtr, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostGetFlags(unsigned int * flagsPtr, void * hostPtr)
{
	__ORO_FUNC(
		CU4ORO::hipHostGetFlags(__ORO_FORCE_CAST(unsigned int *,flagsPtr), __ORO_FORCE_CAST(void *,hostPtr)),
		hipHostGetFlags(flagsPtr, hostPtr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostMalloc(void ** ptr, size_t size, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipHostMalloc(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(unsigned int,flags)),
		hipHostMalloc(ptr, size, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostRegister(void * hostPtr, size_t sizeBytes, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipHostRegister(__ORO_FORCE_CAST(void *,hostPtr), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(unsigned int,flags)),
		hipHostRegister(hostPtr, sizeBytes, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroHostUnregister(void * hostPtr)
{
	__ORO_FUNC(
		CU4ORO::hipHostUnregister(__ORO_FORCE_CAST(void *,hostPtr)),
		hipHostUnregister(hostPtr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroImportExternalMemory(oroExternalMemory_t * extMem_out, const oroExternalMemoryHandleDesc * memHandleDesc)
{
	__ORO_FUNC(
		CU4ORO::hipImportExternalMemory(__ORO_FORCE_CAST(CU4ORO::hipExternalMemory_t *,extMem_out), __ORO_FORCE_CAST(const CU4ORO::hipExternalMemoryHandleDesc *,memHandleDesc)),
		hipImportExternalMemory(extMem_out, memHandleDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroImportExternalSemaphore(oroExternalSemaphore_t * extSem_out, const oroExternalSemaphoreHandleDesc * semHandleDesc)
{
	__ORO_FUNC(
		CU4ORO::hipImportExternalSemaphore(__ORO_FORCE_CAST(CU4ORO::hipExternalSemaphore_t *,extSem_out), __ORO_FORCE_CAST(const CU4ORO::hipExternalSemaphoreHandleDesc *,semHandleDesc)),
		hipImportExternalSemaphore(extSem_out, semHandleDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroIpcCloseMemHandle(void * devPtr)
{
	__ORO_FUNC(
		CU4ORO::hipIpcCloseMemHandle(__ORO_FORCE_CAST(void *,devPtr)),
		hipIpcCloseMemHandle(devPtr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroIpcGetEventHandle(oroIpcEventHandle_t * handle, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::hipIpcGetEventHandle(__ORO_FORCE_CAST(CU4ORO::hipIpcEventHandle_t *,handle), __ORO_FORCE_CAST(CU4ORO::hipEvent_t,event)),
		hipIpcGetEventHandle(handle, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroIpcGetMemHandle(oroIpcMemHandle_t * handle, void * devPtr)
{
	__ORO_FUNC(
		CU4ORO::hipIpcGetMemHandle(__ORO_FORCE_CAST(CU4ORO::hipIpcMemHandle_t *,handle), __ORO_FORCE_CAST(void *,devPtr)),
		hipIpcGetMemHandle(handle, devPtr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroIpcOpenEventHandle(oroEvent_t * event, oroIpcEventHandle_t handle)
{
	__ORO_FUNC(
		CU4ORO::hipIpcOpenEventHandle(__ORO_FORCE_CAST(CU4ORO::hipEvent_t *,event), __ORO_FORCE_CAST(CU4ORO::hipIpcEventHandle_t,handle)),
		hipIpcOpenEventHandle(event, handle)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroIpcOpenMemHandle(void ** devPtr, oroIpcMemHandle_t handle, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipIpcOpenMemHandle(__ORO_FORCE_CAST(void **,devPtr), __ORO_FORCE_CAST(CU4ORO::hipIpcMemHandle_t,handle), __ORO_FORCE_CAST(unsigned int,flags)),
		hipIpcOpenMemHandle(devPtr, handle, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroLaunchCooperativeKernel(const void * f, dim3 gridDim, dim3 blockDimX, void ** kernelParams, unsigned int sharedMemBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipLaunchCooperativeKernel(__ORO_FORCE_CAST(const void *,f), __ORO_FORCE_CAST(CU4ORO::dim3,gridDim), __ORO_FORCE_CAST(CU4ORO::dim3,blockDimX), __ORO_FORCE_CAST(void **,kernelParams), __ORO_FORCE_CAST(unsigned int,sharedMemBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipLaunchCooperativeKernel(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroLaunchCooperativeKernelMultiDevice(oroLaunchParams * launchParamsList, int numDevices, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipLaunchCooperativeKernelMultiDevice(__ORO_FORCE_CAST(CU4ORO::hipLaunchParams *,launchParamsList), __ORO_FORCE_CAST(int,numDevices), __ORO_FORCE_CAST(unsigned int,flags)),
		hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void ** args, size_t sharedMemBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipLaunchKernel(__ORO_FORCE_CAST(const void *,function_address), __ORO_FORCE_CAST(CU4ORO::dim3,numBlocks), __ORO_FORCE_CAST(CU4ORO::dim3,dimBlocks), __ORO_FORCE_CAST(void **,args), __ORO_FORCE_CAST(size_t,sharedMemBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMalloc(void ** ptr, size_t size)
{
	__ORO_FUNC(
		CU4ORO::hipMalloc(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t,size)),
		hipMalloc(ptr, size)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMalloc3D(oroPitchedPtr * pitchedDevPtr, oroExtent extent)
{
	__ORO_FUNC(
		CU4ORO::hipMalloc3D(__ORO_FORCE_CAST(CU4ORO::hipPitchedPtr *,pitchedDevPtr), __ORO_FORCE_CAST(CU4ORO::hipExtent,extent)),
		hipMalloc3D(pitchedDevPtr, extent)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMalloc3DArray(oroArray_t * array, const  oroChannelFormatDesc * desc,  oroExtent extent, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMalloc3DArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t *,array), __ORO_FORCE_CAST(const CU4ORO::hipChannelFormatDesc *,desc), __ORO_FORCE_CAST(CU4ORO::hipExtent,extent), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMalloc3DArray(array, desc, extent, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocArray(oroArray_t * array, const oroChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMallocArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t *,array), __ORO_FORCE_CAST(const CU4ORO::hipChannelFormatDesc *,desc), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMallocArray(array, desc, width, height, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocAsync(void ** dev_ptr, size_t size, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMallocAsync(__ORO_FORCE_CAST(void **,dev_ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMallocAsync(dev_ptr, size, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocFromPoolAsync(void ** dev_ptr, size_t size, oroMemPool_t mem_pool, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMallocFromPoolAsync(__ORO_FORCE_CAST(void **,dev_ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocHost(void ** ptr, size_t size)
{
	__ORO_FUNC(
		CU4ORO::hipMallocHost(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t,size)),
		hipMallocHost(ptr, size)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocManaged(void ** dev_ptr, size_t size, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMallocManaged(__ORO_FORCE_CAST(void **,dev_ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMallocManaged(dev_ptr, size, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocMipmappedArray(oroMipmappedArray_t * mipmappedArray, const  oroChannelFormatDesc * desc,  oroExtent extent, unsigned int numLevels, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMallocMipmappedArray(__ORO_FORCE_CAST(CU4ORO::hipMipmappedArray_t *,mipmappedArray), __ORO_FORCE_CAST(const CU4ORO::hipChannelFormatDesc *,desc), __ORO_FORCE_CAST(CU4ORO::hipExtent,extent), __ORO_FORCE_CAST(unsigned int,numLevels), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMallocPitch(void ** ptr, size_t * pitch, size_t width, size_t height)
{
	__ORO_FUNC(
		CU4ORO::hipMallocPitch(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t *,pitch), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height)),
		hipMallocPitch(ptr, pitch, width, height)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemAddressFree(void * devPtr, size_t size)
{
	__ORO_FUNC(
		CU4ORO::hipMemAddressFree(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,devPtr), __ORO_FORCE_CAST(size_t,size)),
		hipMemAddressFree(devPtr, size)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemAddressReserve(void ** ptr, size_t size, size_t alignment, void * addr, unsigned long long flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemAddressReserve(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t *,ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(size_t,alignment), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,addr), __ORO_FORCE_CAST(unsigned long long,flags)),
		hipMemAddressReserve(ptr, size, alignment, addr, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemAdvise(const void * dev_ptr, size_t count, oroMemoryAdvise advice, int device)
{
	__ORO_FUNC(
		CU4ORO::hipMemAdvise(__ORO_FORCE_CAST(const void *,dev_ptr), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipMemoryAdvise,advice), __ORO_FORCE_CAST(int,device)),
		hipMemAdvise(dev_ptr, count, advice, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemAllocHost(void ** ptr, size_t size)
{
	__ORO_FUNC(
		CU4ORO::hipMemAllocHost(__ORO_FORCE_CAST(void **,ptr), __ORO_FORCE_CAST(size_t,size)),
		hipMemAllocHost(ptr, size)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemAllocPitch(oroDeviceptr_t * dptr, size_t * pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemAllocPitch(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t *,dptr), __ORO_FORCE_CAST(size_t *,pitch), __ORO_FORCE_CAST(size_t,widthInBytes), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(unsigned int,elementSizeBytes)),
		hipMemAllocPitch(dptr, pitch, widthInBytes, height, elementSizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemCreate(oroMemGenericAllocationHandle_t * handle, size_t size, const oroMemAllocationProp * prop, unsigned long long flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemCreate(__ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle *,handle), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(const CU4ORO::hipMemAllocationProp *,prop), __ORO_FORCE_CAST(unsigned long long,flags)),
		hipMemCreate(handle, size, prop, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemExportToShareableHandle(void * shareableHandle, oroMemGenericAllocationHandle_t handle, oroMemAllocationHandleType handleType, unsigned long long flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemExportToShareableHandle(__ORO_FORCE_CAST(void *,shareableHandle), __ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle,handle), __ORO_FORCE_CAST(CU4ORO::hipMemAllocationHandleType,handleType), __ORO_FORCE_CAST(unsigned long long,flags)),
		hipMemExportToShareableHandle(shareableHandle, handle, handleType, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemGetAccess(unsigned long long * flags, const oroMemLocation * location, void * ptr)
{
	__ORO_FUNC(
		CU4ORO::hipMemGetAccess(__ORO_FORCE_CAST(unsigned long long *,flags), __ORO_FORCE_CAST(const CU4ORO::hipMemLocation *,location), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr)),
		hipMemGetAccess(flags, location, ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemGetAddressRange(oroDeviceptr_t * pbase, size_t * psize, oroDeviceptr_t dptr)
{
	__ORO_FUNC(
		CU4ORO::hipMemGetAddressRange(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t *,pbase), __ORO_FORCE_CAST(size_t *,psize), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dptr)),
		hipMemGetAddressRange(pbase, psize, dptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemGetAllocationGranularity(size_t * granularity, const oroMemAllocationProp * prop, oroMemAllocationGranularity_flags option)
{
	__ORO_FUNC(
		CU4ORO::hipMemGetAllocationGranularity(__ORO_FORCE_CAST(size_t *,granularity), __ORO_FORCE_CAST(const CU4ORO::hipMemAllocationProp *,prop), __ORO_FORCE_CAST(CU4ORO::hipMemAllocationGranularity_flags,option)),
		hipMemGetAllocationGranularity(granularity, prop, option)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemGetAllocationPropertiesFromHandle(oroMemAllocationProp * prop, oroMemGenericAllocationHandle_t handle)
{
	__ORO_FUNC(
		CU4ORO::hipMemGetAllocationPropertiesFromHandle(__ORO_FORCE_CAST(CU4ORO::hipMemAllocationProp *,prop), __ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle,handle)),
		hipMemGetAllocationPropertiesFromHandle(prop, handle)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemGetInfo(size_t * free, size_t * total)
{
	__ORO_FUNC(
		CU4ORO::hipMemGetInfo(__ORO_FORCE_CAST(size_t *,free), __ORO_FORCE_CAST(size_t *,total)),
		hipMemGetInfo(free, total)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemImportFromShareableHandle(oroMemGenericAllocationHandle_t * handle, void * osHandle, oroMemAllocationHandleType shHandleType)
{
	__ORO_FUNC(
		CU4ORO::hipMemImportFromShareableHandle(__ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle *,handle), __ORO_FORCE_CAST(void *,osHandle), __ORO_FORCE_CAST(CU4ORO::hipMemAllocationHandleType,shHandleType)),
		hipMemImportFromShareableHandle(handle, osHandle, shHandleType)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemMap(void * ptr, size_t size, size_t offset, oroMemGenericAllocationHandle_t handle, unsigned long long flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemMap(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(size_t,offset), __ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle,handle), __ORO_FORCE_CAST(unsigned long long,flags)),
		hipMemMap(ptr, size, offset, handle, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemMapArrayAsync(oroArrayMapInfo * mapInfoList, unsigned int count, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemMapArrayAsync(__ORO_FORCE_CAST(CU4ORO::hipArrayMapInfo *,mapInfoList), __ORO_FORCE_CAST(unsigned int,count), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemMapArrayAsync(mapInfoList, count, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolCreate(oroMemPool_t * mem_pool, const oroMemPoolProps * pool_props)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolCreate(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t *,mem_pool), __ORO_FORCE_CAST(const CU4ORO::hipMemPoolProps *,pool_props)),
		hipMemPoolCreate(mem_pool, pool_props)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolDestroy(oroMemPool_t mem_pool)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolDestroy(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool)),
		hipMemPoolDestroy(mem_pool)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolExportPointer(oroMemPoolPtrExportData * export_data, void * dev_ptr)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolExportPointer(__ORO_FORCE_CAST(CU4ORO::hipMemPoolPtrExportData *,export_data), __ORO_FORCE_CAST(void *,dev_ptr)),
		hipMemPoolExportPointer(export_data, dev_ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolExportToShareableHandle(void * shared_handle, oroMemPool_t mem_pool, oroMemAllocationHandleType handle_type, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolExportToShareableHandle(__ORO_FORCE_CAST(void *,shared_handle), __ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipMemAllocationHandleType,handle_type), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMemPoolExportToShareableHandle(shared_handle, mem_pool, handle_type, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolGetAccess(oroMemAccessFlags * flags, oroMemPool_t mem_pool, oroMemLocation * location)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolGetAccess(__ORO_FORCE_CAST(CU4ORO::hipMemAccessFlags *,flags), __ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipMemLocation *,location)),
		hipMemPoolGetAccess(flags, mem_pool, location)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolGetAttribute(oroMemPool_t mem_pool, oroMemPoolAttr attr, void * value)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolGetAttribute(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipMemPoolAttr,attr), __ORO_FORCE_CAST(void *,value)),
		hipMemPoolGetAttribute(mem_pool, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolImportFromShareableHandle(oroMemPool_t * mem_pool, void * shared_handle, oroMemAllocationHandleType handle_type, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolImportFromShareableHandle(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t *,mem_pool), __ORO_FORCE_CAST(void *,shared_handle), __ORO_FORCE_CAST(CU4ORO::hipMemAllocationHandleType,handle_type), __ORO_FORCE_CAST(unsigned int,flags)),
		hipMemPoolImportFromShareableHandle(mem_pool, shared_handle, handle_type, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolImportPointer(void ** dev_ptr, oroMemPool_t mem_pool, oroMemPoolPtrExportData * export_data)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolImportPointer(__ORO_FORCE_CAST(void **,dev_ptr), __ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipMemPoolPtrExportData *,export_data)),
		hipMemPoolImportPointer(dev_ptr, mem_pool, export_data)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolSetAccess(oroMemPool_t mem_pool, const oroMemAccessDesc * desc_list, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolSetAccess(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(const CU4ORO::hipMemAccessDesc *,desc_list), __ORO_FORCE_CAST(size_t,count)),
		hipMemPoolSetAccess(mem_pool, desc_list, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolSetAttribute(oroMemPool_t mem_pool, oroMemPoolAttr attr, void * value)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolSetAttribute(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(CU4ORO::hipMemPoolAttr,attr), __ORO_FORCE_CAST(void *,value)),
		hipMemPoolSetAttribute(mem_pool, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPoolTrimTo(oroMemPool_t mem_pool, size_t min_bytes_to_hold)
{
	__ORO_FUNC(
		CU4ORO::hipMemPoolTrimTo(__ORO_FORCE_CAST(CU4ORO::hipMemPool_t,mem_pool), __ORO_FORCE_CAST(size_t,min_bytes_to_hold)),
		hipMemPoolTrimTo(mem_pool, min_bytes_to_hold)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemPrefetchAsync(const void * dev_ptr, size_t count, int device, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemPrefetchAsync(__ORO_FORCE_CAST(const void *,dev_ptr), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(int,device), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemPrefetchAsync(dev_ptr, count, device, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemRangeGetAttribute(void * data, size_t data_size, oroMemRangeAttribute attribute, const void * dev_ptr, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemRangeGetAttribute(__ORO_FORCE_CAST(void *,data), __ORO_FORCE_CAST(size_t,data_size), __ORO_FORCE_CAST(CU4ORO::hipMemRangeAttribute,attribute), __ORO_FORCE_CAST(const void *,dev_ptr), __ORO_FORCE_CAST(size_t,count)),
		hipMemRangeGetAttribute(data, data_size, attribute, dev_ptr, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemRangeGetAttributes(void ** data, size_t * data_sizes, oroMemRangeAttribute * attributes, size_t num_attributes, const void * dev_ptr, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemRangeGetAttributes(__ORO_FORCE_CAST(void **,data), __ORO_FORCE_CAST(size_t *,data_sizes), __ORO_FORCE_CAST(CU4ORO::hipMemRangeAttribute *,attributes), __ORO_FORCE_CAST(size_t,num_attributes), __ORO_FORCE_CAST(const void *,dev_ptr), __ORO_FORCE_CAST(size_t,count)),
		hipMemRangeGetAttributes(data, data_sizes, attributes, num_attributes, dev_ptr, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemRelease(oroMemGenericAllocationHandle_t handle)
{
	__ORO_FUNC(
		CU4ORO::hipMemRelease(__ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle,handle)),
		hipMemRelease(handle)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemRetainAllocationHandle(oroMemGenericAllocationHandle_t * handle, void * addr)
{
	__ORO_FUNC(
		CU4ORO::hipMemRetainAllocationHandle(__ORO_FORCE_CAST(CU4ORO::CUmemGenericAllocationHandle *,handle), __ORO_FORCE_CAST(void *,addr)),
		hipMemRetainAllocationHandle(handle, addr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemSetAccess(void * ptr, size_t size, const oroMemAccessDesc * desc, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemSetAccess(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr), __ORO_FORCE_CAST(size_t,size), __ORO_FORCE_CAST(const CU4ORO::hipMemAccessDesc *,desc), __ORO_FORCE_CAST(size_t,count)),
		hipMemSetAccess(ptr, size, desc, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemUnmap(void * ptr, size_t size)
{
	__ORO_FUNC(
		CU4ORO::hipMemUnmap(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr), __ORO_FORCE_CAST(size_t,size)),
		hipMemUnmap(ptr, size)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy(void * dst, const void * src, size_t sizeBytes, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpy(dst, src, sizeBytes, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2D(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,dpitch), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,spitch), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2DAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,dpitch), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,spitch), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2DFromArray(void * dst, size_t dpitch, oroArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2DFromArray(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,dpitch), __ORO_FORCE_CAST(CU4ORO::hipArray_t,src), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2DFromArrayAsync(void * dst, size_t dpitch, oroArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2DFromArrayAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,dpitch), __ORO_FORCE_CAST(CU4ORO::hipArray_t,src), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2DToArray(oroArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2DToArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t,dst), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,spitch), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy2DToArrayAsync(oroArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy2DToArrayAsync(__ORO_FORCE_CAST(CU4ORO::hipArray_t,dst), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,spitch), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy3D(const  oroMemcpy3DParms * p)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy3D(__ORO_FORCE_CAST(const struct CU4ORO::cudaMemcpy3DParms *,p)),
		hipMemcpy3D(p)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpy3DAsync(const  oroMemcpy3DParms * p, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpy3DAsync(__ORO_FORCE_CAST(const struct CU4ORO::cudaMemcpy3DParms *,p), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpy3DAsync(p, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyAsync(void * dst, const void * src, size_t sizeBytes, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyAsync(dst, src, sizeBytes, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyAtoH(void * dst, oroArray_t srcArray, size_t srcOffset, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyAtoH(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(CU4ORO::hipArray_t,srcArray), __ORO_FORCE_CAST(size_t,srcOffset), __ORO_FORCE_CAST(size_t,count)),
		hipMemcpyAtoH(dst, srcArray, srcOffset, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyDtoD(oroDeviceptr_t dst, oroDeviceptr_t src, size_t sizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyDtoD(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dst), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,src), __ORO_FORCE_CAST(size_t,sizeBytes)),
		hipMemcpyDtoD(dst, src, sizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyDtoDAsync(oroDeviceptr_t dst, oroDeviceptr_t src, size_t sizeBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyDtoDAsync(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dst), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyDtoDAsync(dst, src, sizeBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyDtoH(void * dst, oroDeviceptr_t src, size_t sizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyDtoH(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,src), __ORO_FORCE_CAST(size_t,sizeBytes)),
		hipMemcpyDtoH(dst, src, sizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyDtoHAsync(void * dst, oroDeviceptr_t src, size_t sizeBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyDtoHAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyDtoHAsync(dst, src, sizeBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyFromArray(void * dst, oroArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyFromArray(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(CU4ORO::hipArray_const_t,srcArray), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpyFromArray(dst, srcArray, wOffset, hOffset, count, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyFromSymbol(void * dst, const void * symbol, size_t sizeBytes, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyFromSymbol(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(const void *,symbol), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(size_t,offset), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t sizeBytes, size_t offset, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyFromSymbolAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(const void *,symbol), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(size_t,offset), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyHtoA(oroArray_t dstArray, size_t dstOffset, const void * srcHost, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyHtoA(__ORO_FORCE_CAST(CU4ORO::hipArray_t,dstArray), __ORO_FORCE_CAST(size_t,dstOffset), __ORO_FORCE_CAST(const void *,srcHost), __ORO_FORCE_CAST(size_t,count)),
		hipMemcpyHtoA(dstArray, dstOffset, srcHost, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyHtoD(oroDeviceptr_t dst, void * src, size_t sizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyHtoD(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dst), __ORO_FORCE_CAST(void *,src), __ORO_FORCE_CAST(size_t,sizeBytes)),
		hipMemcpyHtoD(dst, src, sizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyHtoDAsync(oroDeviceptr_t dst, void * src, size_t sizeBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyHtoDAsync(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dst), __ORO_FORCE_CAST(void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyHtoDAsync(dst, src, sizeBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyParam2D(const oro_Memcpy2D * pCopy)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyParam2D(__ORO_FORCE_CAST(const CU4ORO::hip_Memcpy2D *,pCopy)),
		hipMemcpyParam2D(pCopy)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyParam2DAsync(const oro_Memcpy2D * pCopy, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyParam2DAsync(__ORO_FORCE_CAST(const CU4ORO::hip_Memcpy2D *,pCopy), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyParam2DAsync(pCopy, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyPeer(void * dst, int dstDeviceId, const void * src, int srcDeviceId, size_t sizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyPeer(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(int,dstDeviceId), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(int,srcDeviceId), __ORO_FORCE_CAST(size_t,sizeBytes)),
		hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyPeerAsync(void * dst, int dstDeviceId, const void * src, int srcDevice, size_t sizeBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyPeerAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(int,dstDeviceId), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(int,srcDevice), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyToArray(oroArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyToArray(__ORO_FORCE_CAST(CU4ORO::hipArray_t,dst), __ORO_FORCE_CAST(size_t,wOffset), __ORO_FORCE_CAST(size_t,hOffset), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpyToArray(dst, wOffset, hOffset, src, count, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyToSymbol(const void * symbol, const void * src, size_t sizeBytes, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyToSymbol(__ORO_FORCE_CAST(const void *,symbol), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(size_t,offset), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind)),
		hipMemcpyToSymbol(symbol, src, sizeBytes, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyToSymbolAsync(const void * symbol, const void * src, size_t sizeBytes, size_t offset, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyToSymbolAsync(__ORO_FORCE_CAST(const void *,symbol), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(size_t,offset), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemcpyWithStream(void * dst, const void * src, size_t sizeBytes, oroMemcpyKind kind, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemcpyWithStream(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(const void *,src), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipMemcpyKind,kind), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemcpyWithStream(dst, src, sizeBytes, kind, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemset(void * dst, int value, size_t sizeBytes)
{
	__ORO_FUNC(
		CU4ORO::hipMemset(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,sizeBytes)),
		hipMemset(dst, value, sizeBytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemset2D(void * dst, size_t pitch, int value, size_t width, size_t height)
{
	__ORO_FUNC(
		CU4ORO::hipMemset2D(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,pitch), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height)),
		hipMemset2D(dst, pitch, value, width, height)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemset2DAsync(void * dst, size_t pitch, int value, size_t width, size_t height, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemset2DAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(size_t,pitch), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,width), __ORO_FORCE_CAST(size_t,height), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemset2DAsync(dst, pitch, value, width, height, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemset3D(oroPitchedPtr pitchedDevPtr, int value, oroExtent extent)
{
	__ORO_FUNC(
		CU4ORO::hipMemset3D(__ORO_FORCE_CAST(CU4ORO::hipPitchedPtr,pitchedDevPtr), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(CU4ORO::hipExtent,extent)),
		hipMemset3D(pitchedDevPtr, value, extent)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemset3DAsync(oroPitchedPtr pitchedDevPtr, int value, oroExtent extent, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemset3DAsync(__ORO_FORCE_CAST(CU4ORO::hipPitchedPtr,pitchedDevPtr), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(CU4ORO::hipExtent,extent), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemset3DAsync(pitchedDevPtr, value, extent, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetAsync(void * dst, int value, size_t sizeBytes, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetAsync(__ORO_FORCE_CAST(void *,dst), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,sizeBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemsetAsync(dst, value, sizeBytes, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD16(oroDeviceptr_t dest, unsigned short value, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD16(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dest), __ORO_FORCE_CAST(unsigned short,value), __ORO_FORCE_CAST(size_t,count)),
		hipMemsetD16(dest, value, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD16Async(oroDeviceptr_t dest, unsigned short value, size_t count, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD16Async(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dest), __ORO_FORCE_CAST(unsigned short,value), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemsetD16Async(dest, value, count, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD32(oroDeviceptr_t dest, int value, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD32(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dest), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,count)),
		hipMemsetD32(dest, value, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD32Async(oroDeviceptr_t dst, int value, size_t count, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD32Async(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dst), __ORO_FORCE_CAST(int,value), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemsetD32Async(dst, value, count, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD8(oroDeviceptr_t dest, unsigned char value, size_t count)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD8(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dest), __ORO_FORCE_CAST(unsigned char,value), __ORO_FORCE_CAST(size_t,count)),
		hipMemsetD8(dest, value, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMemsetD8Async(oroDeviceptr_t dest, unsigned char value, size_t count, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipMemsetD8Async(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,dest), __ORO_FORCE_CAST(unsigned char,value), __ORO_FORCE_CAST(size_t,count), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipMemsetD8Async(dest, value, count, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMipmappedArrayCreate(oroMipmappedArray_t * pHandle, ORO_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int numMipmapLevels)
{
	__ORO_FUNC(
		CU4ORO::hipMipmappedArrayCreate(__ORO_FORCE_CAST(CU4ORO::hipmipmappedArray *,pHandle), __ORO_FORCE_CAST(CU4ORO::CUDA_ARRAY3D_DESCRIPTOR *,pMipmappedArrayDesc), __ORO_FORCE_CAST(unsigned int,numMipmapLevels)),
		hipMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMipmappedArrayDestroy(oroMipmappedArray_t hMipmappedArray)
{
	__ORO_FUNC(
		CU4ORO::hipMipmappedArrayDestroy(__ORO_FORCE_CAST(CU4ORO::hipmipmappedArray,hMipmappedArray)),
		hipMipmappedArrayDestroy(hMipmappedArray)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroMipmappedArrayGetLevel(oroArray_t * pLevelArray, oroMipmappedArray_t hMipMappedArray, unsigned int level)
{
	__ORO_FUNC(
		CU4ORO::hipMipmappedArrayGetLevel(__ORO_FORCE_CAST(CU4ORO::hipArray_t *,pLevelArray), __ORO_FORCE_CAST(CU4ORO::hipmipmappedArray,hMipMappedArray), __ORO_FORCE_CAST(unsigned int,level)),
		hipMipmappedArrayGetLevel(pLevelArray, hMipMappedArray, level)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleGetFunction(oroFunction_t * function, oroModule_t module, const char * kname)
{
	__ORO_FUNC(
		CU4ORO::hipModuleGetFunction(__ORO_FORCE_CAST(CU4ORO::hipFunction_t *,function), __ORO_FORCE_CAST(CU4ORO::hipModule_t,module), __ORO_FORCE_CAST(const char *,kname)),
		hipModuleGetFunction(function, module, kname)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleGetGlobal(oroDeviceptr_t * dptr, size_t * bytes, oroModule_t hmod, const char * name)
{
	__ORO_FUNC(
		CU4ORO::hipModuleGetGlobal(__ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t *,dptr), __ORO_FORCE_CAST(size_t *,bytes), __ORO_FORCE_CAST(CU4ORO::hipModule_t,hmod), __ORO_FORCE_CAST(const char *,name)),
		hipModuleGetGlobal(dptr, bytes, hmod, name)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleGetTexRef(textureReference ** texRef, oroModule_t hmod, const char * name)
{
	__ORO_FUNC(
		CU4ORO::hipModuleGetTexRef(__ORO_FORCE_CAST(CU4ORO::CUtexref *,texRef), __ORO_FORCE_CAST(CU4ORO::hipModule_t,hmod), __ORO_FORCE_CAST(const char *,name)),
		hipModuleGetTexRef(texRef, hmod, name)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLaunchCooperativeKernel(oroFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, oroStream_t stream, void ** kernelParams)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLaunchCooperativeKernel(__ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(unsigned int,gridDimX), __ORO_FORCE_CAST(unsigned int,gridDimY), __ORO_FORCE_CAST(unsigned int,gridDimZ), __ORO_FORCE_CAST(unsigned int,blockDimX), __ORO_FORCE_CAST(unsigned int,blockDimY), __ORO_FORCE_CAST(unsigned int,blockDimZ), __ORO_FORCE_CAST(unsigned int,sharedMemBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(void **,kernelParams)),
		hipModuleLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLaunchCooperativeKernelMultiDevice(oroFunctionLaunchParams * launchParamsList, unsigned int numDevices, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLaunchCooperativeKernelMultiDevice(__ORO_FORCE_CAST(CU4ORO::hipFunctionLaunchParams *,launchParamsList), __ORO_FORCE_CAST(unsigned int,numDevices), __ORO_FORCE_CAST(unsigned int,flags)),
		hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLaunchKernel(oroFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, oroStream_t stream, void ** kernelParams, void ** extra)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLaunchKernel(__ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(unsigned int,gridDimX), __ORO_FORCE_CAST(unsigned int,gridDimY), __ORO_FORCE_CAST(unsigned int,gridDimZ), __ORO_FORCE_CAST(unsigned int,blockDimX), __ORO_FORCE_CAST(unsigned int,blockDimY), __ORO_FORCE_CAST(unsigned int,blockDimZ), __ORO_FORCE_CAST(unsigned int,sharedMemBytes), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(void **,kernelParams), __ORO_FORCE_CAST(void **,extra)),
		hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLoad(oroModule_t * module, const char * fname)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLoad(__ORO_FORCE_CAST(CU4ORO::hipModule_t *,module), __ORO_FORCE_CAST(const char *,fname)),
		hipModuleLoad(module, fname)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLoadData(oroModule_t * module, const void * image)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLoadData(__ORO_FORCE_CAST(CU4ORO::hipModule_t *,module), __ORO_FORCE_CAST(const void *,image)),
		hipModuleLoadData(module, image)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleLoadDataEx(oroModule_t * module, const void * image, unsigned int numOptions, oroJitOption * options, void ** optionValues)
{
	__ORO_FUNC(
		CU4ORO::hipModuleLoadDataEx(__ORO_FORCE_CAST(CU4ORO::hipModule_t *,module), __ORO_FORCE_CAST(const void *,image), __ORO_FORCE_CAST(unsigned int,numOptions), __ORO_FORCE_CAST(CU4ORO::hipJitOption *,options), __ORO_FORCE_CAST(void **,optionValues)),
		hipModuleLoadDataEx(module, image, numOptions, options, optionValues)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, oroFunction_t f, int blockSize, size_t dynSharedMemPerBlk)
{
	__ORO_FUNC(
		CU4ORO::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(__ORO_FORCE_CAST(int *,numBlocks), __ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(int,blockSize), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk)),
		hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, oroFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(__ORO_FORCE_CAST(int *,numBlocks), __ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(int,blockSize), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk), __ORO_FORCE_CAST(unsigned int,flags)),
		hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize, dynSharedMemPerBlk, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, oroFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit)
{
	__ORO_FUNC(
		CU4ORO::hipModuleOccupancyMaxPotentialBlockSize(__ORO_FORCE_CAST(int *,gridSize), __ORO_FORCE_CAST(int *,blockSize), __ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk), __ORO_FORCE_CAST(int,blockSizeLimit)),
		hipModuleOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize, int * blockSize, oroFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipModuleOccupancyMaxPotentialBlockSizeWithFlags(__ORO_FORCE_CAST(int *,gridSize), __ORO_FORCE_CAST(int *,blockSize), __ORO_FORCE_CAST(CU4ORO::hipFunction_t,f), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk), __ORO_FORCE_CAST(int,blockSizeLimit), __ORO_FORCE_CAST(unsigned int,flags)),
		hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroModuleUnload(oroModule_t module)
{
	__ORO_FUNC(
		CU4ORO::hipModuleUnload(__ORO_FORCE_CAST(CU4ORO::hipModule_t,module)),
		hipModuleUnload(module)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk)
{
	__ORO_FUNC(
		CU4ORO::hipOccupancyMaxActiveBlocksPerMultiprocessor(__ORO_FORCE_CAST(int *,numBlocks), __ORO_FORCE_CAST(const void *,f), __ORO_FORCE_CAST(int,blockSize), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk)),
		hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(__ORO_FORCE_CAST(int *,numBlocks), __ORO_FORCE_CAST(const void *,f), __ORO_FORCE_CAST(int,blockSize), __ORO_FORCE_CAST(size_t,dynSharedMemPerBlk), __ORO_FORCE_CAST(unsigned int,flags)),
		hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize, dynSharedMemPerBlk, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroPeekAtLastError()
{
	__ORO_FUNC(
		CU4ORO::hipPeekAtLastError(),
		hipPeekAtLastError()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroPointerGetAttribute(void * data, oroPointer_attribute attribute, oroDeviceptr_t ptr)
{
	__ORO_FUNC(
		CU4ORO::hipPointerGetAttribute(__ORO_FORCE_CAST(void *,data), __ORO_FORCE_CAST(CU4ORO::CUpointer_attribute,attribute), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t,ptr)),
		hipPointerGetAttribute(data, attribute, ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroPointerGetAttributes(oroPointerAttribute_t * attributes, const void * ptr)
{
	__ORO_FUNC(
		CU4ORO::hipPointerGetAttributes(__ORO_FORCE_CAST(CU4ORO::hipPointerAttribute_t *,attributes), __ORO_FORCE_CAST(const void *,ptr)),
		hipPointerGetAttributes(attributes, ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroProfilerStart()
{
	__ORO_FUNC(
		CU4ORO::hipProfilerStart(),
		hipProfilerStart()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroProfilerStop()
{
	__ORO_FUNC(
		CU4ORO::hipProfilerStop(),
		hipProfilerStop()     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroRuntimeGetVersion(int * runtimeVersion)
{
	__ORO_FUNC(
		CU4ORO::hipRuntimeGetVersion(__ORO_FORCE_CAST(int *,runtimeVersion)),
		hipRuntimeGetVersion(runtimeVersion)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroSetDevice(int deviceId)
{
	__ORO_FUNC(
		CU4ORO::hipSetDevice(__ORO_FORCE_CAST(int,deviceId)),
		hipSetDevice(deviceId)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroSetDeviceFlags(unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipSetDeviceFlags(__ORO_FORCE_CAST(unsigned int,flags)),
		hipSetDeviceFlags(flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroSignalExternalSemaphoresAsync(const oroExternalSemaphore_t * extSemArray, const oroExternalSemaphoreSignalParams * paramsArray, unsigned int numExtSems, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipSignalExternalSemaphoresAsync(__ORO_FORCE_CAST(const CU4ORO::hipExternalSemaphore_t *,extSemArray), __ORO_FORCE_CAST(const CU4ORO::hipExternalSemaphoreSignalParams *,paramsArray), __ORO_FORCE_CAST(unsigned int,numExtSems), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamAddCallback(oroStream_t stream, oroStreamCallback_t callback, void * userData, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipStreamAddCallback(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(CU4ORO::hipStreamCallback_t,callback), __ORO_FORCE_CAST(void *,userData), __ORO_FORCE_CAST(unsigned int,flags)),
		hipStreamAddCallback(stream, callback, userData, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamAttachMemAsync(oroStream_t stream, void * dev_ptr, size_t length, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipStreamAttachMemAsync(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(CU4ORO::hipDeviceptr_t *,dev_ptr), __ORO_FORCE_CAST(size_t,length), __ORO_FORCE_CAST(unsigned int,flags)),
		hipStreamAttachMemAsync(stream, dev_ptr, length, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamCreate(oroStream_t * stream)
{
	__ORO_FUNC(
		CU4ORO::hipStreamCreate(__ORO_FORCE_CAST(CU4ORO::hipStream_t *,stream)),
		hipStreamCreate(stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamCreateWithFlags(oroStream_t * stream, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipStreamCreateWithFlags(__ORO_FORCE_CAST(CU4ORO::hipStream_t *,stream), __ORO_FORCE_CAST(unsigned int,flags)),
		hipStreamCreateWithFlags(stream, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamCreateWithPriority(oroStream_t * stream, unsigned int flags, int priority)
{
	__ORO_FUNC(
		CU4ORO::hipStreamCreateWithPriority(__ORO_FORCE_CAST(CU4ORO::hipStream_t *,stream), __ORO_FORCE_CAST(unsigned int,flags), __ORO_FORCE_CAST(int,priority)),
		hipStreamCreateWithPriority(stream, flags, priority)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamDestroy(oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipStreamDestroy(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipStreamDestroy(stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamGetDevice(oroStream_t stream, oroDevice_t * device)
{
	__ORO_FUNC(
		CU4ORO::hipStreamGetDevice(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(CU4ORO::hipDevice_t *,device)),
		hipStreamGetDevice(stream, device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamGetFlags(oroStream_t stream, unsigned int * flags)
{
	__ORO_FUNC(
		CU4ORO::hipStreamGetFlags(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(unsigned int *,flags)),
		hipStreamGetFlags(stream, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamGetPriority(oroStream_t stream, int * priority)
{
	__ORO_FUNC(
		CU4ORO::hipStreamGetPriority(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(int *,priority)),
		hipStreamGetPriority(stream, priority)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamQuery(oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipStreamQuery(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipStreamQuery(stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamSynchronize(oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipStreamSynchronize(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipStreamSynchronize(stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamWaitEvent(oroStream_t stream, oroEvent_t event, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::hipStreamWaitEvent(__ORO_FORCE_CAST(CU4ORO::hipStream_t,stream), __ORO_FORCE_CAST(CU4ORO::hipEvent_t,event), __ORO_FORCE_CAST(unsigned int,flags)),
		hipStreamWaitEvent(stream, event, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroWaitExternalSemaphoresAsync(const oroExternalSemaphore_t * extSemArray, const oroExternalSemaphoreWaitParams * paramsArray, unsigned int numExtSems, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::hipWaitExternalSemaphoresAsync(__ORO_FORCE_CAST(const CU4ORO::hipExternalSemaphore_t *,extSemArray), __ORO_FORCE_CAST(const CU4ORO::hipExternalSemaphoreWaitParams *,paramsArray), __ORO_FORCE_CAST(unsigned int,numExtSems), __ORO_FORCE_CAST(CU4ORO::hipStream_t,stream)),
		hipWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)     );
	return oroErrorUnknown;
}
orortcResult OROAPI orortcAddNameExpression(orortcProgram prog, const char * name_expression)
{
	__ORO_FUNC(
		CU4ORO::hiprtcAddNameExpression(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(const char *,name_expression)),
		hiprtcAddNameExpression(prog, name_expression)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCompileProgram(orortcProgram prog, int numOptions, const char ** options)
{
	__ORO_FUNC(
		CU4ORO::hiprtcCompileProgram(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(int,numOptions), __ORO_FORCE_CAST(const char **,options)),
		hiprtcCompileProgram(prog, numOptions, options)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcCreateProgram(orortcProgram * prog, const char * src, const char * name, int numHeaders, const char ** headers, const char ** includeNames)
{
	__ORO_FUNC(
		CU4ORO::hiprtcCreateProgram(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram *,prog), __ORO_FORCE_CAST(const char *,src), __ORO_FORCE_CAST(const char *,name), __ORO_FORCE_CAST(int,numHeaders), __ORO_FORCE_CAST(const char **,headers), __ORO_FORCE_CAST(const char **,includeNames)),
		hiprtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcDestroyProgram(orortcProgram * prog)
{
	__ORO_FUNC(
		CU4ORO::hiprtcDestroyProgram(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram *,prog)),
		hiprtcDestroyProgram(prog)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCode(orortcProgram prog, char * code)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetCode(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(char *,code)),
		hiprtcGetCode(prog, code)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetCodeSize(orortcProgram prog, size_t * codeSizeRet)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetCodeSize(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(size_t *,codeSizeRet)),
		hiprtcGetCodeSize(prog, codeSizeRet)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
const char * OROAPI orortcGetErrorString(orortcResult result)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetErrorString(__ORO_FORCE_CAST(CU4ORO::hiprtcResult,result)),
		hiprtcGetErrorString(result)     );
	return nullptr;
}
orortcResult OROAPI orortcGetLoweredName(orortcProgram prog, const char * name_expression, const char ** lowered_name)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetLoweredName(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(const char *,name_expression), __ORO_FORCE_CAST(const char **,lowered_name)),
		hiprtcGetLoweredName(prog, name_expression, lowered_name)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetProgramLog(orortcProgram prog, char * log)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetProgramLog(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(char *,log)),
		hiprtcGetProgramLog(prog, log)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcGetProgramLogSize(orortcProgram prog, size_t * logSizeRet)
{
	__ORO_FUNC(
		CU4ORO::hiprtcGetProgramLogSize(__ORO_FORCE_CAST(CU4ORO::hiprtcProgram,prog), __ORO_FORCE_CAST(size_t *,logSizeRet)),
		hiprtcGetProgramLogSize(prog, logSizeRet)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcVersion(int * major, int * minor)
{
	__ORO_FUNC(
		CU4ORO::hiprtcVersion(__ORO_FORCE_CAST(int *,major), __ORO_FORCE_CAST(int *,minor)),
		hiprtcVersion(major, minor)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
oroError_t OROAPI oroArray3DCreate(oroArray_t * array, const ORO_ARRAY3D_DESCRIPTOR * pAllocateArray)
{
	__ORO_FUNC(
		CU4ORO::cuArray3DCreate_v2((CU4ORO::CUarray *)array, (const CU4ORO::CUDA_ARRAY3D_DESCRIPTOR *)pAllocateArray),
		hipArray3DCreate(array, pAllocateArray)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroArray3DGetDescriptor(ORO_ARRAY3D_DESCRIPTOR * pArrayDescriptor, oroArray_t array)
{
	__ORO_FUNC(
		CU4ORO::cuArray3DGetDescriptor_v2((CU4ORO::CUDA_ARRAY3D_DESCRIPTOR *)pArrayDescriptor, (CU4ORO::CUarray)array),
		hipArray3DGetDescriptor(pArrayDescriptor, array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroArrayCreate(oroArray_t * pHandle, const ORO_ARRAY_DESCRIPTOR * pAllocateArray)
{
	__ORO_FUNC(
		CU4ORO::cuArrayCreate_v2((CU4ORO::CUarray *)pHandle, (const CU4ORO::CUDA_ARRAY_DESCRIPTOR *)pAllocateArray),
		hipArrayCreate(pHandle, pAllocateArray)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroArrayDestroy(oroArray_t array)
{
	__ORO_FUNC(
		CU4ORO::cuArrayDestroy((CU4ORO::CUarray)array),
		hipArrayDestroy(array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroArrayGetDescriptor(ORO_ARRAY_DESCRIPTOR * pArrayDescriptor, oroArray_t array)
{
	__ORO_FUNC(
		CU4ORO::cuArrayGetDescriptor_v2((CU4ORO::CUDA_ARRAY_DESCRIPTOR *)pArrayDescriptor, (CU4ORO::CUarray)array),
		hipArrayGetDescriptor(pArrayDescriptor, array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroArrayGetInfo(oroChannelFormatDesc * desc, oroExtent * extent, unsigned int * flags, oroArray_t array)
{
	__ORO_FUNC(
		CU4ORO::cudaArrayGetInfo((struct CU4ORO::cudaChannelFormatDesc *)desc, (struct CU4ORO::cudaExtent *)extent, (unsigned int *)flags, (CU4ORO::cudaArray_t)array),
		hipArrayGetInfo(desc, extent, flags, array)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGetGraphMemAttribute(int device, oroGraphMemAttributeType attr, void * value)
{
	__ORO_FUNC(
		CU4ORO::cudaDeviceGetGraphMemAttribute((int)device, (enum CU4ORO::cudaGraphMemAttributeType)attr, (void *)value),
		hipDeviceGetGraphMemAttribute(device, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceGraphMemTrim(int device)
{
	__ORO_FUNC(
		CU4ORO::cudaDeviceGraphMemTrim((int)device),
		hipDeviceGraphMemTrim(device)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDeviceSetGraphMemAttribute(int device, oroGraphMemAttributeType attr, void * value)
{
	__ORO_FUNC(
		CU4ORO::cudaDeviceSetGraphMemAttribute((int)device, (enum CU4ORO::cudaGraphMemAttributeType)attr, (void *)value),
		hipDeviceSetGraphMemAttribute(device, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvGraphAddMemcpyNode(oroGraphNode_t * phGraphNode, oroGraph_t hGraph, const oroGraphNode_t * dependencies, size_t numDependencies, const ORO_MEMCPY3D * copyParams, oroCtx_t ctx)
{
	__ORO_FUNC(
		CU4ORO::cuGraphAddMemcpyNode((CU4ORO::CUgraphNode *)phGraphNode, (CU4ORO::CUgraph)hGraph, (const CU4ORO::CUgraphNode *)dependencies, (size_t)numDependencies, (const CU4ORO::CUDA_MEMCPY3D *)copyParams, (CU4ORO::CUcontext)ctx),
		hipDrvGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroDrvMemcpy2DUnaligned(const oro_Memcpy2D * pCopy)
{
	__ORO_FUNC(
		CU4ORO::cuMemcpy2DUnaligned_v2((const CU4ORO::CUDA_MEMCPY2D *)pCopy),
		hipDrvMemcpy2DUnaligned(pCopy)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetTextureObjectResourceViewDesc( oroResourceViewDesc * pResViewDesc, oroTextureObject_t textureObject)
{
	__ORO_FUNC(
		CU4ORO::cudaGetTextureObjectResourceViewDesc((struct CU4ORO::cudaResourceViewDesc *)pResViewDesc, (CU4ORO::cudaTextureObject_t)textureObject),
		hipGetTextureObjectResourceViewDesc(pResViewDesc, textureObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGetTextureObjectTextureDesc(oroTextureDesc * pTexDesc, oroTextureObject_t textureObject)
{
	__ORO_FUNC(
		CU4ORO::cudaGetTextureObjectTextureDesc((struct CU4ORO::cudaTextureDesc *)pTexDesc, (CU4ORO::cudaTextureObject_t)textureObject),
		hipGetTextureObjectTextureDesc(pTexDesc, textureObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddChildGraphNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, oroGraph_t childGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddChildGraphNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (CU4ORO::cudaGraph_t)childGraph),
		hipGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddDependencies(oroGraph_t graph, const oroGraphNode_t * from, const oroGraphNode_t * to, size_t numDependencies)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddDependencies((CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)from, (const CU4ORO::cudaGraphNode_t *)to, (size_t)numDependencies),
		hipGraphAddDependencies(graph, from, to, numDependencies)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddEmptyNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddEmptyNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies),
		hipGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddEventRecordNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddEventRecordNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (CU4ORO::cudaEvent_t)event),
		hipGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddEventWaitNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddEventWaitNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (CU4ORO::cudaEvent_t)event),
		hipGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddHostNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, const oroHostNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddHostNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (const struct CU4ORO::cudaHostNodeParams *)pNodeParams),
		hipGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddKernelNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, const oroKernelNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddKernelNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (const struct CU4ORO::cudaKernelNodeParams *)pNodeParams),
		hipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemAllocNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, oroMemAllocNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemAllocNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (struct CU4ORO::cudaMemAllocNodeParams *)pNodeParams),
		hipGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemFreeNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, void * dev_ptr)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemFreeNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (void *)dev_ptr),
		hipGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dev_ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemcpyNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, const oroMemcpy3DParms * pCopyParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemcpyNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (const struct CU4ORO::cudaMemcpy3DParms *)pCopyParams),
		hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemcpyNode1D(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemcpyNode1D((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (void *)dst, (const void *)src, (size_t)count, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemcpyNodeFromSymbol(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemcpyNodeFromSymbol((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (void *)dst, (const void *)symbol, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemcpyNodeToSymbol(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemcpyNodeToSymbol((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (const void *)symbol, (const void *)src, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphAddMemsetNode(oroGraphNode_t * pGraphNode, oroGraph_t graph, const oroGraphNode_t * pDependencies, size_t numDependencies, const oroMemsetParams * pMemsetParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphAddMemsetNode((CU4ORO::cudaGraphNode_t *)pGraphNode, (CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)pDependencies, (size_t)numDependencies, (const struct CU4ORO::cudaMemsetParams *)pMemsetParams),
		hipGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphChildGraphNodeGetGraph(oroGraphNode_t node, oroGraph_t * pGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphChildGraphNodeGetGraph((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaGraph_t *)pGraph),
		hipGraphChildGraphNodeGetGraph(node, pGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphClone(oroGraph_t * pGraphClone, oroGraph_t originalGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphClone((CU4ORO::cudaGraph_t *)pGraphClone, (CU4ORO::cudaGraph_t)originalGraph),
		hipGraphClone(pGraphClone, originalGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphCreate(oroGraph_t * pGraph, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphCreate((CU4ORO::cudaGraph_t *)pGraph, (unsigned int)flags),
		hipGraphCreate(pGraph, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphDebugDotPrint(oroGraph_t graph, const char * path, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphDebugDotPrint((CU4ORO::cudaGraph_t)graph, (const char *)path, (unsigned int)flags),
		hipGraphDebugDotPrint(graph, path, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphDestroy(oroGraph_t graph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphDestroy((CU4ORO::cudaGraph_t)graph),
		hipGraphDestroy(graph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphDestroyNode(oroGraphNode_t node)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphDestroyNode((CU4ORO::cudaGraphNode_t)node),
		hipGraphDestroyNode(node)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphEventRecordNodeGetEvent(oroGraphNode_t node, oroEvent_t * event_out)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphEventRecordNodeGetEvent((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaEvent_t *)event_out),
		hipGraphEventRecordNodeGetEvent(node, event_out)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphEventRecordNodeSetEvent(oroGraphNode_t node, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphEventRecordNodeSetEvent((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaEvent_t)event),
		hipGraphEventRecordNodeSetEvent(node, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphEventWaitNodeGetEvent(oroGraphNode_t node, oroEvent_t * event_out)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphEventWaitNodeGetEvent((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaEvent_t *)event_out),
		hipGraphEventWaitNodeGetEvent(node, event_out)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphEventWaitNodeSetEvent(oroGraphNode_t node, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphEventWaitNodeSetEvent((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaEvent_t)event),
		hipGraphEventWaitNodeSetEvent(node, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecChildGraphNodeSetParams(oroGraphExec_t hGraphExec, oroGraphNode_t node, oroGraph_t childGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecChildGraphNodeSetParams((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaGraph_t)childGraph),
		hipGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecDestroy(oroGraphExec_t graphExec)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecDestroy((CU4ORO::cudaGraphExec_t)graphExec),
		hipGraphExecDestroy(graphExec)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecEventRecordNodeSetEvent(oroGraphExec_t hGraphExec, oroGraphNode_t hNode, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecEventRecordNodeSetEvent((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)hNode, (CU4ORO::cudaEvent_t)event),
		hipGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecEventWaitNodeSetEvent(oroGraphExec_t hGraphExec, oroGraphNode_t hNode, oroEvent_t event)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecEventWaitNodeSetEvent((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)hNode, (CU4ORO::cudaEvent_t)event),
		hipGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecHostNodeSetParams(oroGraphExec_t hGraphExec, oroGraphNode_t node, const oroHostNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecHostNodeSetParams((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaHostNodeParams *)pNodeParams),
		hipGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecKernelNodeSetParams(oroGraphExec_t hGraphExec, oroGraphNode_t node, const oroKernelNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecKernelNodeSetParams((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaKernelNodeParams *)pNodeParams),
		hipGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecMemcpyNodeSetParams(oroGraphExec_t hGraphExec, oroGraphNode_t node, oroMemcpy3DParms * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecMemcpyNodeSetParams((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaMemcpy3DParms *)pNodeParams),
		hipGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecMemcpyNodeSetParams1D(oroGraphExec_t hGraphExec, oroGraphNode_t node, void * dst, const void * src, size_t count, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecMemcpyNodeSetParams1D((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (void *)dst, (const void *)src, (size_t)count, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecMemcpyNodeSetParamsFromSymbol(oroGraphExec_t hGraphExec, oroGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecMemcpyNodeSetParamsFromSymbol((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (void *)dst, (const void *)symbol, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecMemcpyNodeSetParamsToSymbol(oroGraphExec_t hGraphExec, oroGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecMemcpyNodeSetParamsToSymbol((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (const void *)symbol, (const void *)src, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphExecMemsetNodeSetParams(oroGraphExec_t hGraphExec, oroGraphNode_t node, const oroMemsetParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphExecMemsetNodeSetParams((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaMemsetParams *)pNodeParams),
		hipGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphGetEdges(oroGraph_t graph, oroGraphNode_t * from, oroGraphNode_t * to, size_t * numEdges)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphGetEdges((CU4ORO::cudaGraph_t)graph, (CU4ORO::cudaGraphNode_t *)from, (CU4ORO::cudaGraphNode_t *)to, (size_t *)numEdges),
		hipGraphGetEdges(graph, from, to, numEdges)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphGetNodes(oroGraph_t graph, oroGraphNode_t * nodes, size_t * numNodes)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphGetNodes((CU4ORO::cudaGraph_t)graph, (CU4ORO::cudaGraphNode_t *)nodes, (size_t *)numNodes),
		hipGraphGetNodes(graph, nodes, numNodes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphGetRootNodes(oroGraph_t graph, oroGraphNode_t * pRootNodes, size_t * pNumRootNodes)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphGetRootNodes((CU4ORO::cudaGraph_t)graph, (CU4ORO::cudaGraphNode_t *)pRootNodes, (size_t *)pNumRootNodes),
		hipGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphHostNodeGetParams(oroGraphNode_t node, oroHostNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphHostNodeGetParams((CU4ORO::cudaGraphNode_t)node, (struct CU4ORO::cudaHostNodeParams *)pNodeParams),
		hipGraphHostNodeGetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphHostNodeSetParams(oroGraphNode_t node, const oroHostNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphHostNodeSetParams((CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaHostNodeParams *)pNodeParams),
		hipGraphHostNodeSetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphInstantiateWithFlags(oroGraphExec_t * pGraphExec, oroGraph_t graph, unsigned long long flags)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphInstantiateWithFlags((CU4ORO::cudaGraphExec_t *)pGraphExec, (CU4ORO::cudaGraph_t)graph, (unsigned long long)flags),
		hipGraphInstantiateWithFlags(pGraphExec, graph, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphKernelNodeCopyAttributes(oroGraphNode_t hSrc, oroGraphNode_t hDst)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphKernelNodeCopyAttributes((CU4ORO::cudaGraphNode_t)hSrc, (CU4ORO::cudaGraphNode_t)hDst),
		hipGraphKernelNodeCopyAttributes(hSrc, hDst)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphKernelNodeGetAttribute(oroGraphNode_t hNode, oroKernelNodeAttrID attr, oroKernelNodeAttrValue * value)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphKernelNodeGetAttribute((CU4ORO::cudaGraphNode_t)hNode, (CU4ORO::cudaLaunchAttributeID)attr, (CU4ORO::cudaLaunchAttributeValue *)value),
		hipGraphKernelNodeGetAttribute(hNode, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphKernelNodeGetParams(oroGraphNode_t node, oroKernelNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphKernelNodeGetParams((CU4ORO::cudaGraphNode_t)node, (struct CU4ORO::cudaKernelNodeParams *)pNodeParams),
		hipGraphKernelNodeGetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphKernelNodeSetAttribute(oroGraphNode_t hNode, oroKernelNodeAttrID attr, const oroKernelNodeAttrValue * value)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphKernelNodeSetAttribute((CU4ORO::cudaGraphNode_t)hNode, (CU4ORO::cudaLaunchAttributeID)attr, (const CU4ORO::cudaLaunchAttributeValue *)value),
		hipGraphKernelNodeSetAttribute(hNode, attr, value)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphKernelNodeSetParams(oroGraphNode_t node, const oroKernelNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphKernelNodeSetParams((CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaKernelNodeParams *)pNodeParams),
		hipGraphKernelNodeSetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphLaunch(oroGraphExec_t graphExec, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphLaunch((CU4ORO::cudaGraphExec_t)graphExec, (CU4ORO::cudaStream_t)stream),
		hipGraphLaunch(graphExec, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemAllocNodeGetParams(oroGraphNode_t node, oroMemAllocNodeParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemAllocNodeGetParams((CU4ORO::cudaGraphNode_t)node, (struct CU4ORO::cudaMemAllocNodeParams *)pNodeParams),
		hipGraphMemAllocNodeGetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemFreeNodeGetParams(oroGraphNode_t node, void * dev_ptr)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemFreeNodeGetParams((CU4ORO::cudaGraphNode_t)node, (void *)dev_ptr),
		hipGraphMemFreeNodeGetParams(node, dev_ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemcpyNodeGetParams(oroGraphNode_t node, oroMemcpy3DParms * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemcpyNodeGetParams((CU4ORO::cudaGraphNode_t)node, (struct CU4ORO::cudaMemcpy3DParms *)pNodeParams),
		hipGraphMemcpyNodeGetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemcpyNodeSetParams(oroGraphNode_t node, const oroMemcpy3DParms * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemcpyNodeSetParams((CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaMemcpy3DParms *)pNodeParams),
		hipGraphMemcpyNodeSetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemcpyNodeSetParams1D(oroGraphNode_t node, void * dst, const void * src, size_t count, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemcpyNodeSetParams1D((CU4ORO::cudaGraphNode_t)node, (void *)dst, (const void *)src, (size_t)count, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemcpyNodeSetParamsFromSymbol(oroGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemcpyNodeSetParamsFromSymbol((CU4ORO::cudaGraphNode_t)node, (void *)dst, (const void *)symbol, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemcpyNodeSetParamsToSymbol(oroGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, oroMemcpyKind kind)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemcpyNodeSetParamsToSymbol((CU4ORO::cudaGraphNode_t)node, (const void *)symbol, (const void *)src, (size_t)count, (size_t)offset, (enum CU4ORO::cudaMemcpyKind)kind),
		hipGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemsetNodeGetParams(oroGraphNode_t node, oroMemsetParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemsetNodeGetParams((CU4ORO::cudaGraphNode_t)node, (struct CU4ORO::cudaMemsetParams *)pNodeParams),
		hipGraphMemsetNodeGetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphMemsetNodeSetParams(oroGraphNode_t node, const oroMemsetParams * pNodeParams)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphMemsetNodeSetParams((CU4ORO::cudaGraphNode_t)node, (const struct CU4ORO::cudaMemsetParams *)pNodeParams),
		hipGraphMemsetNodeSetParams(node, pNodeParams)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeFindInClone(oroGraphNode_t * pNode, oroGraphNode_t originalNode, oroGraph_t clonedGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeFindInClone((CU4ORO::cudaGraphNode_t *)pNode, (CU4ORO::cudaGraphNode_t)originalNode, (CU4ORO::cudaGraph_t)clonedGraph),
		hipGraphNodeFindInClone(pNode, originalNode, clonedGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeGetDependencies(oroGraphNode_t node, oroGraphNode_t * pDependencies, size_t * pNumDependencies)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeGetDependencies((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaGraphNode_t *)pDependencies, (size_t *)pNumDependencies),
		hipGraphNodeGetDependencies(node, pDependencies, pNumDependencies)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeGetDependentNodes(oroGraphNode_t node, oroGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeGetDependentNodes((CU4ORO::cudaGraphNode_t)node, (CU4ORO::cudaGraphNode_t *)pDependentNodes, (size_t *)pNumDependentNodes),
		hipGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeGetEnabled(oroGraphExec_t hGraphExec, oroGraphNode_t hNode, unsigned int * isEnabled)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeGetEnabled((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)hNode, (unsigned int *)isEnabled),
		hipGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeGetType(oroGraphNode_t node, oroGraphNodeType * pType)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeGetType((CU4ORO::cudaGraphNode_t)node, (enum CU4ORO::cudaGraphNodeType *)pType),
		hipGraphNodeGetType(node, pType)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphNodeSetEnabled(oroGraphExec_t hGraphExec, oroGraphNode_t hNode, unsigned int isEnabled)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphNodeSetEnabled((CU4ORO::cudaGraphExec_t)hGraphExec, (CU4ORO::cudaGraphNode_t)hNode, (unsigned int)isEnabled),
		hipGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphReleaseUserObject(oroGraph_t graph, oroUserObject_t object, unsigned int count)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphReleaseUserObject((CU4ORO::cudaGraph_t)graph, (CU4ORO::cudaUserObject_t)object, (unsigned int)count),
		hipGraphReleaseUserObject(graph, object, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphRemoveDependencies(oroGraph_t graph, const oroGraphNode_t * from, const oroGraphNode_t * to, size_t numDependencies)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphRemoveDependencies((CU4ORO::cudaGraph_t)graph, (const CU4ORO::cudaGraphNode_t *)from, (const CU4ORO::cudaGraphNode_t *)to, (size_t)numDependencies),
		hipGraphRemoveDependencies(graph, from, to, numDependencies)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphRetainUserObject(oroGraph_t graph, oroUserObject_t object, unsigned int count, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphRetainUserObject((CU4ORO::cudaGraph_t)graph, (CU4ORO::cudaUserObject_t)object, (unsigned int)count, (unsigned int)flags),
		hipGraphRetainUserObject(graph, object, count, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroGraphUpload(oroGraphExec_t graphExec, oroStream_t stream)
{
	__ORO_FUNC(
		CU4ORO::cudaGraphUpload((CU4ORO::cudaGraphExec_t)graphExec, (CU4ORO::cudaStream_t)stream),
		hipGraphUpload(graphExec, stream)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroLaunchHostFunc(oroStream_t stream, oroHostFn_t fn, void * userData)
{
	__ORO_FUNC(
		CU4ORO::cudaLaunchHostFunc((CU4ORO::cudaStream_t)stream, (CU4ORO::cudaHostFn_t)fn, (void *)userData),
		hipLaunchHostFunc(stream, fn, userData)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroPointerSetAttribute(const void * value, oroPointer_attribute attribute, oroDeviceptr_t ptr)
{
	__ORO_FUNC(
		CU4ORO::cuPointerSetAttribute((const void *)value, (CU4ORO::CUpointer_attribute)attribute, (CU4ORO::CUdeviceptr)ptr),
		hipPointerSetAttribute(value, attribute, ptr)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamBeginCapture(oroStream_t stream, oroStreamCaptureMode mode)
{
	__ORO_FUNC(
		CU4ORO::cudaStreamBeginCapture((CU4ORO::cudaStream_t)stream, (enum CU4ORO::cudaStreamCaptureMode)mode),
		hipStreamBeginCapture(stream, mode)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamEndCapture(oroStream_t stream, oroGraph_t * pGraph)
{
	__ORO_FUNC(
		CU4ORO::cudaStreamEndCapture((CU4ORO::cudaStream_t)stream, (CU4ORO::cudaGraph_t *)pGraph),
		hipStreamEndCapture(stream, pGraph)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamGetCaptureInfo_v2(oroStream_t stream, oroStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, oroGraph_t * graph_out, const oroGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
	__ORO_FUNC(
		CU4ORO::cudaStreamGetCaptureInfo_v2((CU4ORO::cudaStream_t)stream, (enum CU4ORO::cudaStreamCaptureStatus *)captureStatus_out, (unsigned long long *)id_out, (CU4ORO::cudaGraph_t *)graph_out, (const CU4ORO::cudaGraphNode_t **)dependencies_out, (size_t *)numDependencies_out),
		hipStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamIsCapturing(oroStream_t stream, oroStreamCaptureStatus * pCaptureStatus)
{
	__ORO_FUNC(
		CU4ORO::cudaStreamIsCapturing((CU4ORO::cudaStream_t)stream, (enum CU4ORO::cudaStreamCaptureStatus *)pCaptureStatus),
		hipStreamIsCapturing(stream, pCaptureStatus)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroStreamUpdateCaptureDependencies(oroStream_t stream, oroGraphNode_t * dependencies, size_t numDependencies, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cudaStreamUpdateCaptureDependencies((CU4ORO::cudaStream_t)stream, (CU4ORO::cudaGraphNode_t *)dependencies, (size_t)numDependencies, (unsigned int)flags),
		hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexObjectCreate(oroTextureObject_t * pTexObject, const ORO_RESOURCE_DESC * pResDesc, const ORO_TEXTURE_DESC * pTexDesc, const ORO_RESOURCE_VIEW_DESC * pResViewDesc)
{
	__ORO_FUNC(
		CU4ORO::cuTexObjectCreate((CU4ORO::CUtexObject *)pTexObject, (const CU4ORO::CUDA_RESOURCE_DESC *)pResDesc, (const CU4ORO::CUDA_TEXTURE_DESC *)pTexDesc, (const CU4ORO::CUDA_RESOURCE_VIEW_DESC *)pResViewDesc),
		hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexObjectDestroy(oroTextureObject_t texObject)
{
	__ORO_FUNC(
		CU4ORO::cuTexObjectDestroy((CU4ORO::CUtexObject)texObject),
		hipTexObjectDestroy(texObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexObjectGetResourceDesc(ORO_RESOURCE_DESC * pResDesc, oroTextureObject_t texObject)
{
	__ORO_FUNC(
		CU4ORO::cuTexObjectGetResourceDesc((CU4ORO::CUDA_RESOURCE_DESC *)pResDesc, (CU4ORO::CUtexObject)texObject),
		hipTexObjectGetResourceDesc(pResDesc, texObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexObjectGetResourceViewDesc(ORO_RESOURCE_VIEW_DESC * pResViewDesc, oroTextureObject_t texObject)
{
	__ORO_FUNC(
		CU4ORO::cuTexObjectGetResourceViewDesc((CU4ORO::CUDA_RESOURCE_VIEW_DESC *)pResViewDesc, (CU4ORO::CUtexObject)texObject),
		hipTexObjectGetResourceViewDesc(pResViewDesc, texObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexObjectGetTextureDesc(ORO_TEXTURE_DESC * pTexDesc, oroTextureObject_t texObject)
{
	__ORO_FUNC(
		CU4ORO::cuTexObjectGetTextureDesc((CU4ORO::CUDA_TEXTURE_DESC *)pTexDesc, (CU4ORO::CUtexObject)texObject),
		hipTexObjectGetTextureDesc(pTexDesc, texObject)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetAddress(oroDeviceptr_t * dev_ptr, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetAddress_v2((CU4ORO::CUdeviceptr *)dev_ptr, (CU4ORO::CUtexref)texRef),
		hipTexRefGetAddress(dev_ptr, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetAddressMode( oroTextureAddressMode * pam, const textureReference * texRef, int dim)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetAddressMode((CU4ORO::CUaddress_mode *)pam, (CU4ORO::CUtexref)texRef, (int)dim),
		hipTexRefGetAddressMode(pam, texRef, dim)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetFilterMode( oroTextureFilterMode * pfm, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetFilterMode((CU4ORO::CUfilter_mode *)pfm, (CU4ORO::CUtexref)texRef),
		hipTexRefGetFilterMode(pfm, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetFlags(unsigned int * pFlags, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetFlags((unsigned int *)pFlags, (CU4ORO::CUtexref)texRef),
		hipTexRefGetFlags(pFlags, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetFormat(oroArray_Format * pFormat, int * pNumChannels, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetFormat((CU4ORO::CUarray_format *)pFormat, (int *)pNumChannels, (CU4ORO::CUtexref)texRef),
		hipTexRefGetFormat(pFormat, pNumChannels, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetMaxAnisotropy(int * pmaxAnsio, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetMaxAnisotropy((int *)pmaxAnsio, (CU4ORO::CUtexref)texRef),
		hipTexRefGetMaxAnisotropy(pmaxAnsio, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetMipmapFilterMode( oroTextureFilterMode * pfm, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetMipmapFilterMode((CU4ORO::CUfilter_mode *)pfm, (CU4ORO::CUtexref)texRef),
		hipTexRefGetMipmapFilterMode(pfm, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetMipmapLevelBias(float * pbias, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetMipmapLevelBias((float *)pbias, (CU4ORO::CUtexref)texRef),
		hipTexRefGetMipmapLevelBias(pbias, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, const textureReference * texRef)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefGetMipmapLevelClamp((float *)pminMipmapLevelClamp, (float *)pmaxMipmapLevelClamp, (CU4ORO::CUtexref)texRef),
		hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetAddress(size_t * ByteOffset, textureReference * texRef, oroDeviceptr_t dptr, size_t bytes)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetAddress_v2((size_t *)ByteOffset, (CU4ORO::CUtexref)texRef, (CU4ORO::CUdeviceptr)dptr, (size_t)bytes),
		hipTexRefSetAddress(ByteOffset, texRef, dptr, bytes)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetAddress2D(textureReference * texRef, const ORO_ARRAY_DESCRIPTOR * desc, oroDeviceptr_t dptr, size_t Pitch)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetAddress2D_v3((CU4ORO::CUtexref)texRef, (const CU4ORO::CUDA_ARRAY_DESCRIPTOR *)desc, (CU4ORO::CUdeviceptr)dptr, (size_t)Pitch),
		hipTexRefSetAddress2D(texRef, desc, dptr, Pitch)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetAddressMode(textureReference * texRef, int dim,  oroTextureAddressMode am)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetAddressMode((CU4ORO::CUtexref)texRef, (int)dim, (CU4ORO::CUaddress_mode)am),
		hipTexRefSetAddressMode(texRef, dim, am)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetArray(textureReference * tex, oroArray_const_t array, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetArray((CU4ORO::CUtexref)tex, (CU4ORO::CUarray)array, (unsigned int)flags),
		hipTexRefSetArray(tex, array, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetBorderColor(textureReference * texRef, float * pBorderColor)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetBorderColor((CU4ORO::CUtexref)texRef, (float *)pBorderColor),
		hipTexRefSetBorderColor(texRef, pBorderColor)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetFilterMode(textureReference * texRef,  oroTextureFilterMode fm)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetFilterMode((CU4ORO::CUtexref)texRef, (CU4ORO::CUfilter_mode)fm),
		hipTexRefSetFilterMode(texRef, fm)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetFlags(textureReference * texRef, unsigned int Flags)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetFlags((CU4ORO::CUtexref)texRef, (unsigned int)Flags),
		hipTexRefSetFlags(texRef, Flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetFormat(textureReference * texRef, oroArray_Format fmt, int NumPackedComponents)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetFormat((CU4ORO::CUtexref)texRef, (CU4ORO::CUarray_format)fmt, (int)NumPackedComponents),
		hipTexRefSetFormat(texRef, fmt, NumPackedComponents)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetMaxAnisotropy(textureReference * texRef, unsigned int maxAniso)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetMaxAnisotropy((CU4ORO::CUtexref)texRef, (unsigned int)maxAniso),
		hipTexRefSetMaxAnisotropy(texRef, maxAniso)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetMipmapFilterMode(textureReference * texRef,  oroTextureFilterMode fm)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetMipmapFilterMode((CU4ORO::CUtexref)texRef, (CU4ORO::CUfilter_mode)fm),
		hipTexRefSetMipmapFilterMode(texRef, fm)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetMipmapLevelBias(textureReference * texRef, float bias)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetMipmapLevelBias((CU4ORO::CUtexref)texRef, (float)bias),
		hipTexRefSetMipmapLevelBias(texRef, bias)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetMipmapLevelClamp(textureReference * texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetMipmapLevelClamp((CU4ORO::CUtexref)texRef, (float)minMipMapLevelClamp, (float)maxMipMapLevelClamp),
		hipTexRefSetMipmapLevelClamp(texRef, minMipMapLevelClamp, maxMipMapLevelClamp)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroTexRefSetMipmappedArray(textureReference * texRef,  oroMipmappedArray * mipmappedArray, unsigned int Flags)
{
	__ORO_FUNC(
		CU4ORO::cuTexRefSetMipmappedArray((CU4ORO::CUtexref)texRef, (CU4ORO::CUmipmappedArray)mipmappedArray, (unsigned int)Flags),
		hipTexRefSetMipmappedArray(texRef, mipmappedArray, Flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroThreadExchangeStreamCaptureMode(oroStreamCaptureMode * mode)
{
	__ORO_FUNC(
		CU4ORO::cudaThreadExchangeStreamCaptureMode((enum CU4ORO::cudaStreamCaptureMode *)mode),
		hipThreadExchangeStreamCaptureMode(mode)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroUserObjectCreate(oroUserObject_t * object_out, void * ptr, oroHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)
{
	__ORO_FUNC(
		CU4ORO::cudaUserObjectCreate((CU4ORO::cudaUserObject_t *)object_out, (void *)ptr, (CU4ORO::cudaHostFn_t)destroy, (unsigned int)initialRefcount, (unsigned int)flags),
		hipUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroUserObjectRelease(oroUserObject_t object, unsigned int count)
{
	__ORO_FUNC(
		CU4ORO::cudaUserObjectRelease((CU4ORO::cudaUserObject_t)object, (unsigned int)count),
		hipUserObjectRelease(object, count)     );
	return oroErrorUnknown;
}
oroError_t OROAPI oroUserObjectRetain(oroUserObject_t object, unsigned int count)
{
	__ORO_FUNC(
		CU4ORO::cudaUserObjectRetain((CU4ORO::cudaUserObject_t)object, (unsigned int)count),
		hipUserObjectRetain(object, count)     );
	return oroErrorUnknown;
}
orortcResult OROAPI orortcLinkAddData(orortcLinkState hip_link_state, orortcJITInputType input_type, void * image, size_t image_size, const char * name, unsigned int num_options, orortcJIT_option * options_ptr, void ** option_values)
{
	__ORO_FUNC(
		cu2nvrtc(CU4ORO::cuLinkAddData_v2((CU4ORO::CUlinkState)hip_link_state, (CU4ORO::CUjitInputType)input_type, (void *)image, (size_t)image_size, (const char *)name, (unsigned int)num_options, (CU4ORO::CUjit_option *)options_ptr, (void **)option_values)),
		hiprtcLinkAddData(hip_link_state, input_type, image, image_size, name, num_options, options_ptr, option_values)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkAddFile(orortcLinkState hip_link_state, orortcJITInputType input_type, const char * file_path, unsigned int num_options, orortcJIT_option * options_ptr, void ** option_values)
{
	__ORO_FUNC(
		cu2nvrtc(CU4ORO::cuLinkAddFile_v2((CU4ORO::CUlinkState)hip_link_state, (CU4ORO::CUjitInputType)input_type, (const char *)file_path, (unsigned int)num_options, (CU4ORO::CUjit_option *)options_ptr, (void **)option_values)),
		hiprtcLinkAddFile(hip_link_state, input_type, file_path, num_options, options_ptr, option_values)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkComplete(orortcLinkState hip_link_state, void ** bin_out, size_t * size_out)
{
	__ORO_FUNC(
		cu2nvrtc(CU4ORO::cuLinkComplete((CU4ORO::CUlinkState)hip_link_state, (void **)bin_out, (size_t *)size_out)),
		hiprtcLinkComplete(hip_link_state, bin_out, size_out)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkCreate(unsigned int num_options, orortcJIT_option * option_ptr, void ** option_vals_pptr, orortcLinkState * hip_link_state_ptr)
{
	__ORO_FUNC(
		cu2nvrtc(CU4ORO::cuLinkCreate_v2((unsigned int)num_options, (CU4ORO::CUjit_option *)option_ptr, (void **)option_vals_pptr, (CU4ORO::CUlinkState *)hip_link_state_ptr)),
		hiprtcLinkCreate(num_options, option_ptr, option_vals_pptr, hip_link_state_ptr)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}
orortcResult OROAPI orortcLinkDestroy(orortcLinkState hip_link_state)
{
	__ORO_FUNC(
		cu2nvrtc(CU4ORO::cuLinkDestroy((CU4ORO::CUlinkState)hip_link_state)),
		hiprtcLinkDestroy(hip_link_state)     );
	return ORORTC_ERROR_INTERNAL_ERROR;
}


///// END REGION: OROCHI_SUMMONER_REGION_orochi_cpp
///// (region automatically generated by Orochi Summoner)
#pragma endregion

