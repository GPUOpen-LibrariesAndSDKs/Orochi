project "SPIRVLink"
   kind "ConsoleApp"

   targetdir "../../dist/bin/%{cfg.buildcfg}"
   location "../../build/"

   if os.istarget("windows") then
      links { "version" }
   else
      links { "dl" }
   end

   includedirs { "../../" }
   files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
   files { "../../contrib/**.h", "../../contrib/**.cpp" }
   files { "main.cpp", "kernels/*.h", "kernels/*.hip", "README.md" }

   filter "files:kernels/*.hip"
      buildaction "None"
   filter {}

   -- The SPIR-V modules are compiled before the executable, by the HIP SDK compiler found through HIP_PATH on
   -- Windows and ROCM_PATH (default /opt/rocm) elsewhere, into <targetdir>/SPIRVLink/ where the test loads them.
   local clang
   if os.istarget("windows") then
      local hipPath = os.getenv("HIP_PATH")
      if hipPath then
         clang = '"' .. path.translate(path.join(hipPath, "bin/clang++.exe")) .. '"'
      else
         premake.warn("SPIRVLink: HIP_PATH is not set, the SPIR-V modules will not be built")
      end
   else
      clang = path.join(os.getenv("ROCM_PATH") or "/opt/rocm", "llvm/bin/clang++")
   end
   if clang then
      local kernels = path.getabsolute("kernels")
      local out = "%{cfg.targetdir}/SPIRVLink"
      prebuildcommands { "{MKDIR} " .. out }
      for _, m in ipairs { "lib", "user", "all" } do
         prebuildcommands {
            clang .. ' -x hip --offload-arch=amdgcnspirv --cuda-device-only -O3 "' .. path.join(kernels, m .. ".hip") ..
            '" -o "' .. out .. "/" .. m .. '.bundle"'
         }
      end
   end
