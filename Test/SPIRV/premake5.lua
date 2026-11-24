project "SPIRV"
   kind "ConsoleApp"

   targetdir "../../dist/bin/%{cfg.buildcfg}"
   location "../../build/"

   if os.istarget("windows") then
      links { "version" }
   end

   includedirs { "../../" }
   files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
   files { "../../contrib/**.h", "../../contrib/**.cpp" }
   files { "main.cpp", "kernel.hip", "README.md" }

   -- Compile the kernel separately with a ROCm compiler supporting amdgcnspirv.
   filter "files:kernel.hip"
      buildaction "None"
   filter {}
