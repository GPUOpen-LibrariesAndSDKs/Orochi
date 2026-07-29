project "simpleD3D12"
    kind "WindowedApp"

    location "%{wks.location}/Test/%{prj.name}"
    debugdir "."

    links { "d3d12", "d3dcompiler", "dxgi" }

    useOrochi()
    linkVersionLib()
    linkWin32SystemLibs()
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
