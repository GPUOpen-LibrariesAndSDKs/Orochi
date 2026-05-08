project "Unittest"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"
 
    filter "system:windows"
        links { "version" }
    filter "system:linux"
        links { "pthread" }
    filter {}

    includedirs { "../" }
    files { "../Orochi/**.h", "../Orochi/**.cpp" }
    files { "*.cpp", "*.h" }
    removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }
    files { "../contrib/**.h", "../contrib/**.cpp" }
    files { "../UnitTest/contrib/**.h", "../UnitTest/contrib/**.cpp" }

    files { "../UnitTest/contrib/gtest-1.6.0/gtest-all.cc" }
    externalincludedirs { "../UnitTest/contrib/gtest-1.6.0/" }
    defines { "GTEST_HAS_TR1_TUPLE=0" }

    if _OPTIONS["kernelcompile"] then
        os.execute("cd ./bitcodes/ && generate_bitcodes.bat")
        os.execute("cd ./bitcodes/ && generate_bitcodes_nvidia.bat")
    end
