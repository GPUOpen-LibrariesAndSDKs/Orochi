project "Unittest"
      kind "ConsoleApp"

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

   if os.istarget("windows") then
      links{ "version" }
   end
   if os.istarget("linux") then
      links { "pthread" }
   end
      includedirs { "../" }
      files { "../Orochi/**.h", "../Orochi/**.cpp" }
      files { "*.cpp", "*.h" }
      removefiles { "moduleTestFunc.cpp", "moduleTestKernel.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }

      files { "../contrib/gtest-1.6.0/gtest-all.cc" }
      sysincludedirs{ "../contrib/gtest-1.6.0/" }
      defines { "GTEST_HAS_TR1_TUPLE=0" }
      if _OPTIONS["kernelcompile"] then
        os.execute( "cd ./bitcodes/ && generate_bitcodes_nvidia.bat" )
      end
