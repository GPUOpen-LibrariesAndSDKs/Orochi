-- Vendored third-party code lives in its own projects so `warnings "Off"`
-- applies at project scope: premake's per-file `warnings` is honoured only by
-- the Visual Studio exporter, so a file filter would leave GCC/Clang unsilenced.
project "gtest"
    kind "StaticLib"
    location "%{wks.location}/contrib/gtest"
    warnings "Off"
    includedirs { "contrib/gtest-1.6.0" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }
    files { "contrib/gtest-1.6.0/**.h", "contrib/gtest-1.6.0/**.cc" }

-- Consumed by the Texture demo, which includes the headers by relative path.
project "stb"
    kind "StaticLib"
    location "%{wks.location}/contrib/stb"
    warnings "Off"
    files { "contrib/stb/**.h", "contrib/stb/**.cpp" }

project "UnitTest"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    useOrochi()
    filter "system:linux"
        links { "pthread" }
    filter {}

    -- Read by demosTest.cpp to locate the demo binaries.
    defines { 'ORO_BUILD_CONFIG="%{cfg.buildcfg}"' }

    files { "*.cpp", "*.h" }
    removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }

    links { "gtest" }
    externalincludedirs { "contrib/gtest-1.6.0/" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    if _OPTIONS["kernelcompile"] then
        local bitcodes = path.getabsolute("bitcodes")
        prebuildScript(bitcodes,
            "generate_bitcodes.bat && generate_bitcodes_nvidia.bat",
            "sh generate_bitcodes.sh && sh generate_bitcodes_nvidia.sh")
    end
