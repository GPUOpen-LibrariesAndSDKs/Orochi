#include <Pop/Pop.h>
#include <contrib/cuew/include/cuew.h>
#include <contrib/hipew/include/hipew.h>

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

#define __PP_FUNC1( cuname, hipname ) if( s_api == API_CUDA ) return cu2pp( cu##cuname ); if( s_api == API_HIP ) return hip2pp( hip##hipname );
#define __PP_FUNC( name ) if( s_api == API_CUDA ) return cu2pp( cu##name ); if( s_api == API_HIP ) return hip2pp( hip##name );
#define __PP_CTXT_FUNC( name ) __PP_FUNC1(Ctx##name, name)
//#define __PP_CTXT_FUNC( name ) if( s_api == API_CUDA ) return cu2pp( cuCtx##name ); if( s_api == API_HIP ) return hip2pp( hip##name );


ppError PPAPI ppGetErrorName(ppError error, const char** pStr)
{
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
//ppError PPAPI ppGetDeviceProperties(ppDeviceProp_t* props, int deviceId);
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
//ppError PPAPI ppDeviceGetAttribute(int* pi, ppDeviceAttribute attrib, ppDe
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