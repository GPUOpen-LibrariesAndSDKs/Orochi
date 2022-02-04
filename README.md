## About

"Yamata No Orochi" [^1] 

A library loads HIP and CUDA driver apis dynamically which allows a user to switch APIs in runtime so you don't need to maintain two similar implementation. 

---

## Requirement

This library does not link to hip nor cuda so you do not need sdks. 
To run the application, you need to install a driver with proper dll. 


----

## To Build

Run premake. 

```
./tools/premake5/win/premake5.exe vs2019
```

Test is a minimum application.

----

## Test Application

It runs on HIP by default. If you want to run on CUDA, run the app with an arg `cuda`. 


----

[^1] Yamata no Orochi (ヤマタノオロチ, 八岐大蛇) is a legendary eight-headed and eight-tailed Japanese dragon.
