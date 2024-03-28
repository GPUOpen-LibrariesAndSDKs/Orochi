project "simpleD3D12"
      kind "WindowedApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"
	  debugdir "."

      buildoptions { "/wd4244" }
      defines { "GTEST_HAS_TR1_TUPLE=0" }
--      links{ "Pop" }
      libdirs{"C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64/"}
      links{"d3d12", "d3dcompiler", "dxgi", "kernel32", "user32", "gdi32", "winspool", "comdlg32", "advapi32", "shell32", "ole32", "oleaut32", "uuid", "odbc32", "odbccp32", "Version"}
      includedirs { "../../" }
      includedirs { "./" }
	  
	  files {"../../contrib/cuew/src/cuew.cpp" }

      files {"../../contrib/hipew/src/hipew.cpp" }
      files {"DX12OroSample.cpp" }
      files {"Main.cpp" }
      files {"../../Orochi/Orochi.cpp" }
      files {"../../Orochi/OrochiUtils.cpp" }
      files {"Win32Application.cpp" }
      files {"simpleD3D12.cpp" }
      files {"stdafx.cpp" }
      files {"../../contrib/cuew/include/cuew.h" }

      files {"../../contrib/hipew/include/hipew.h" }
      files {"DX12OroSample.h" }
      files {"DXSampleHelper.h" }
      files {"helper_string.h" }
      files {"../../Orochi/Orochi.h" }
      files {"../../Orochi/OrochiUtils.h" }
      files {"ShaderStructs.h" }
      files {"Win32Application.h" }
      files {"d3dx12.h" }
      --files {"shaders.hlsl" }
      files {"simpleD3D12.h" }
      files {"stdafx.h" }
	  files{"sinewave_Orochi.oro"}
