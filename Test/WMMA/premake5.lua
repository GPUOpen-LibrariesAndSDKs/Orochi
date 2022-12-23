project "WMMA"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

   if os.istarget("windows") then
      links{ "version" }
   end

      includedirs { "../../" }
      files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "*.h", "*.cpp" }
	  files { "half.hpp" }
