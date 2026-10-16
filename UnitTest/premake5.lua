-- Separate projects because per-file `warnings "Off"` only works in Visual Studio.
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

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

   if os.istarget("windows") then
      links{ "version" }
   end
   if os.istarget("linux") then
      links { "pthread" }
   end
      -- Read by demosTest.cpp to locate the demo binaries; `!` keeps the path absolute.
      defines { 'ORO_DEMO_BIN_DIR="%{!cfg.targetdir}"' }
      includedirs { "../" }
      files { "../Orochi/**.h", "../Orochi/**.cpp" }
      files { "*.cpp", "*.h" }
      removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }

      links { "gtest" }
      externalincludedirs{ "../UnitTest/contrib/gtest-1.6.0/" }
      defines { "GTEST_HAS_TR1_TUPLE=0" }
      if _OPTIONS["kernelcompile"] then
        os.execute( "cd ./bitcodes/ && generate_bitcodes.bat" )
        os.execute( "cd ./bitcodes/ && generate_bitcodes_nvidia.bat" )
      end
