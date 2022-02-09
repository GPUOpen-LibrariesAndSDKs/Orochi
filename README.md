## About

"Yamata No Orochi" [^1] 

Orochi is a library that allows you to load HIP and CUDA APIs dynamically which allows the user to switch APIs at runtime so you don't need to compile two separate implementations for each API. This allows you to compile and maintain a single binary that can run on both AMD and NVIDIA GPU's. Unlike HIP which uses hipamd or CUDA at compile-time, Orochi will dynamically load the corresponding HIP/CUDA shared libraries depending on your platform. In other words, it combines the functionality offered by HIPEW and CUEW into a single library.

---

## Requirement

This library doesn't require you to link to CUDA (for the driver API's) nor HIP (for both driver and runtime API's) at build-time. This provides the benefit that you don't need to install HIP SDK on your machine, or CUDA SDK in case you're not using the runtime API's. 
To run an application compiled with Orochi, you need to install a driver of your choice with the corresponding .dll/.so files based on the GPU(s) available. Orochi will automatically link with the corresponding shared library at runtime.


----

## API example 

APIs have prefix `oro`. If you are familiar with CUDA or HIP driver APIs, you will get used to Orochi APIs easily.  

Here is an example of the API for Orochi device and context creation. 


```
	oroInitialize( ORO_API_HIP, 0 );
	oroInit( 0 );
	oroDevice device;
	oroDeviceGet( &device, 0 );
	oroCtx ctx;
	oroCtxCreate( &ctx, 0, device );
```

See more in the [sample application](./Test/main.cpp).

----

## Building Sample Application

Run premake. 

```
./tools/premake5/win/premake5.exe vs2019
```

Test is a minimum application.

### Test Application

It runs on HIP by default. If you want to run on CUDA, run the app with an arg `cuda`. 

The source code for the application can be found [here](./Test/main.cpp).

----

[^1] Yamata no Orochi (ヤマタノオロチ, 八岐大蛇) is a legendary eight-headed and eight-tailed Japanese dragon.
