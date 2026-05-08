project "simpleD3D12"
    kind "WindowedApp"

    location "%{wks.location}/%{prj.name}"
    debugdir "."

    buildoptions { "/wd4244" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    libdirs { "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64/" }
    links {
        "d3d12", "d3dcompiler", "dxgi",
        "kernel32", "user32", "gdi32", "winspool", "comdlg32",
        "advapi32", "shell32", "ole32", "oleaut32", "uuid",
        "odbc32", "odbccp32", "Version"
    }

    includedirs { "../../" }
    includedirs { "./" }

    files {
        "../../contrib/cuew/src/cuew.cpp",
        "../../contrib/hipew/src/hipew.cpp",
        "DX12OroSample.cpp",
        "Main.cpp",
        "../../Orochi/Orochi.cpp",
        "../../Orochi/OrochiUtils.cpp",
        "Win32Application.cpp",
        "simpleD3D12.cpp",
        "stdafx.cpp",
        "../../contrib/cuew/include/cuew.h",
        "../../contrib/hipew/include/hipew.h",
        "DX12OroSample.h",
        "DXSampleHelper.h",
        "helper_string.h",
        "../../Orochi/Orochi.h",
        "../../Orochi/OrochiUtils.h",
        "ShaderStructs.h",
        "Win32Application.h",
        "d3dx12.h",
        "simpleD3D12.h",
        "stdafx.h",
        "sinewave_Orochi.oro"
    }
