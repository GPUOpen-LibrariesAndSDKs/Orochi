project "SPIRVLink"
   kind "ConsoleApp"

   targetdir "../../dist/bin/%{cfg.buildcfg}"
   location "../../build/"

   links { "dl" }

   includedirs { "../../" }
   files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
   files { "../../contrib/**.h", "../../contrib/**.cpp" }
   files { "main.cpp", "kernels/*.h", "kernels/*.hip", "build_kernels.sh", "README.md" }

   -- The kernels are compiled separately to SPIR-V by build_kernels.sh.
   filter "files:kernels/*.hip"
      buildaction "None"
   filter {}
