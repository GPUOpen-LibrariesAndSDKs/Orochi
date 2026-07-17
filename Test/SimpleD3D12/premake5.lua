project "simpleD3D12"
    kind "WindowedApp"

    location "%{wks.location}/Test/%{prj.name}"
    debugdir "."

    systemversion "latest"
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    links {
        "d3d12", "d3dcompiler", "dxgi",
        "kernel32", "user32", "gdi32", "winspool", "comdlg32",
        "advapi32", "shell32", "ole32", "oleaut32", "uuid",
        "odbc32", "odbccp32", "Version"
    }

    useOrochi()
    includedirs { "./" }

    files {
        "DX12OroSample.cpp",
        "Main.cpp",
        "Win32Application.cpp",
        "simpleD3D12.cpp",
        "stdafx.cpp",
        "DX12OroSample.h",
        "DXSampleHelper.h",
        "helper_string.h",
        "ShaderStructs.h",
        "Win32Application.h",
        "d3dx12.h",
        "simpleD3D12.h",
        "stdafx.h",
        "sinewave_Orochi.oro"
    }
