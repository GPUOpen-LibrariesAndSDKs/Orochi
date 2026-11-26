project "SPIRVLink"
   kind "ConsoleApp"

   targetdir "../../dist/bin/%{cfg.buildcfg}"
   location "../../build/"

   links { "dl" }

   includedirs { "../../" }
   files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
   files { "../../contrib/**.h", "../../contrib/**.cpp" }
   files { "main.cpp", "kernels/*.h", "kernels/*.hip", "spirv/*.spv", "build_kernels.sh", "README.md" }

   -- The kernels are prebuilt (spirv/*.spv); build_kernels.sh regenerates them.
   filter { "files:kernels/*.hip or files:spirv/*.spv" }
      buildaction "None"
   filter {}
