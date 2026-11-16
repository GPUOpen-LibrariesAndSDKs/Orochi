-- Separate projects because per-file `warnings "Off"` only works in Visual Studio.
project "gtest"
    kind "StaticLib"
    location "%{wks.location}/contrib/gtest"
    tags { "thirdparty" }
    includedirs { "contrib/gtest-1.6.0" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }
    files { "contrib/gtest-1.6.0/**.h", "contrib/gtest-1.6.0/**.cc" }

    usage "INTERFACE"
        links { "gtest" }
        externalincludedirs { "contrib/gtest-1.6.0/" }
        defines { "GTEST_HAS_TR1_TUPLE=0" }

-- Consumed by the Texture demo, which includes the headers by relative path.
project "stb"
    kind "StaticLib"
    location "%{wks.location}/contrib/stb"
    tags { "thirdparty" }
    files { "contrib/stb/**.h", "contrib/stb/**.cpp" }

    usage "INTERFACE"
        links { "stb" }

project "UnitTest"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    uses { "Orochi", "gtest" }
    filter "system:linux"
        links { "pthread" }
    filter {}

    -- Read by demosTest.cpp to locate the demo binaries; relative to scripts/, where the tests run.
    local demoBinDir = path.join(
        path.getrelative(path.getabsolute("../scripts"), path.getabsolute(_OPTIONS["distdir"] or "dist")),
        "bin/%{cfg.buildcfg}")
    defines { 'ORO_DEMO_BIN_DIR="' .. demoBinDir .. '"' }

    files { "*.cpp", "*.h" }
    removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }

    if _OPTIONS["kernelcompile"] then
        local bitcodes = path.getabsolute("bitcodes")
        orochiPrebuildScript(bitcodes,
            "generate_bitcodes.bat && generate_bitcodes_nvidia.bat",
            "sh generate_bitcodes.sh && sh generate_bitcodes_nvidia.sh")
    end
