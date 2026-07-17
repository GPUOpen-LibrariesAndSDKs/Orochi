project "UnitTest"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"
 
    useOrochi()
    linkVersionLib()
    filter "system:linux"
        links { "pthread" }
    filter {}

    files { "*.cpp", "*.h" }
    removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }
    files { "contrib/**.h", "contrib/**.cpp" }

    files { "contrib/gtest-1.6.0/gtest-all.cc" }
    externalincludedirs { "contrib/gtest-1.6.0/" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    if _OPTIONS["kernelcompile"] then
        local bitcodes = path.getabsolute("bitcodes")
        if os.ishost("windows") then
            runScript('cd "' .. bitcodes .. '" && generate_bitcodes.bat')
            runScript('cd "' .. bitcodes .. '" && generate_bitcodes_nvidia.bat')
        else
            runScript('cd "' .. bitcodes .. '" && sh generate_bitcodes.sh')
            runScript('cd "' .. bitcodes .. '" && sh generate_bitcodes_nvidia.sh')
        end
    end
