project "UnitTest"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    useOrochi()
    linkVersionLib()
    stageWindowsRuntimeDlls()
    filter "system:linux"
        links { "pthread" }
    filter {}

    -- Read by demosTest.cpp to locate the demo binaries.
    defines { 'ORO_BUILD_CONFIG="%{cfg.buildcfg}"' }

    files { "*.cpp", "*.h" }
    removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }
    files { "contrib/**.h", "contrib/**.cpp", "contrib/**.cc" }

    externalincludedirs { "contrib/gtest-1.6.0/" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    -- Silence vendored gtest/stb so --warning=extra targets only our sources.
    silenceVendoredWarnings("contrib/**")

    if _OPTIONS["kernelcompile"] then
        local bitcodes = path.getabsolute("bitcodes")
        if os.ishost("windows") then
            runScriptIn(bitcodes, "generate_bitcodes.bat")
            runScriptIn(bitcodes, "generate_bitcodes_nvidia.bat")
        else
            runScriptIn(bitcodes, "sh generate_bitcodes.sh")
            runScriptIn(bitcodes, "sh generate_bitcodes_nvidia.sh")
        end
    end
