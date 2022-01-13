#include <Pop/Pop.h>

int main()
{
	int a = ppInitialize( API_HIP, 0 );
	ppError e;
	e = ppInit( 0 );
	ppDevice device;
	e = ppDeviceGet( &device, 0 );
	ppCtx ctx;
	e = ppCtxCreate( &ctx, 0, device );

	return 0;
}
