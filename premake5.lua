-- =============================================================================
-- Orochi (YamatanoOrochi) — Premake5 Build Configuration
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Command-line Options
-- -----------------------------------------------------------------------------

newoption {
   trigger     = "clang",
   description = "Use Clang toolset instead of the default (GCC on Linux, MSVC on Windows)"
}

newoption {
   trigger     = "bakeKernel",
   description = "Bake GPU kernels into source as string literals"
}

newoption {
   trigger     = "precompiled",
   description = "Use precompiled kernels"
}

newoption {
   trigger     = "kernelcompile",
   description = "Compile kernels used for unit test"
}

newoption {
   trigger     = "forceCuda",
   description = "Force CUDA backend even if CUDA_PATH is not found (may cause compilation errors)"
}

newoption {
   trigger     = "builddir",
   value       = "PATH",
   description = "Directory for generated build files (default: build)"
}

newoption {
   trigger     = "warning",
   value       = "LEVEL",
   description = "Compiler warning level: off, on, extra (default: on)",
   allowed     = {
      { "off",   "Disable warnings" },
      { "on",    "Default warnings" },
      { "extra", "Extra warnings" }
   }
}

-- Resolve build output directory (global for subprojects)
buildDir = _OPTIONS["builddir"] or "build"

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

-- Copy files from src_dir to dst_dir with optional glob filter
function copydir(src_dir, dst_dir, filter, single_dst_dir)
   if not os.isdir(src_dir) then
      print("copydir FAILED: " .. src_dir .. " is not an existing directory!")
      return nil
   end

   filter = filter or "**"
   src_dir = src_dir .. "/"
   dst_dir = dst_dir .. "/"

   local dir = path.rebase(".", path.getabsolute("."), src_dir)
   os.chdir(src_dir)
   local matches = os.matchfiles(filter)
   os.chdir(dir)

   local counter = 0
   for _, v in ipairs(matches) do
      local target = iif(single_dst_dir, path.getname(v), v)
      os.mkdir(path.getdirectory(dst_dir .. target))
      if os.copyfile(src_dir .. v, dst_dir .. target) then
         counter = counter + 1
      end
   end

   return counter == #matches or nil
end

-- -----------------------------------------------------------------------------
-- Workspace Definition
-- -----------------------------------------------------------------------------

workspace "YamatanoOrochi"
   configurations { "Debug", "Release" }
   platforms      { "x64" }
   architecture   "x86_64"
   language       "C++"
   cppdialect     "C++20"
   location       (buildDir)
   startproject   "Unittest"

   -- Apply warning level
   local warnLevel = _OPTIONS["warning"] or "on"
   if warnLevel == "off" then
      warnings "Off"
   elseif warnLevel == "extra" then
      warnings "Extra"
   else
      warnings "Default"
   end

   -- Apply Clang toolset if --clang option is specified
   if _OPTIONS["clang"] then
      if os.istarget("windows") then
         toolset "clangcl"
      else
         toolset "clang"
      end
   end

   -- Platform-specific settings
   filter "system:windows"
      defines     { "__WINDOWS__", "_WIN32", "_CRT_SECURE_NO_WARNINGS" }
      characterset "MBCS"
      buildoptions { "/wd4244", "/wd4305", "/wd4018" }

   filter "system:macosx"
      toolset "clang"

   filter "system:linux"
      links { "dl" }

   filter {}

   -- Common defines
   defines { "_CRT_SECURE_NO_WARNINGS" }

   -- Configuration: Debug
   filter { "platforms:x64", "configurations:Debug" }
      targetsuffix "64D"
      defines      { "DEBUG" }
      symbols      "On"
      optimize     "Off"

   -- Configuration: Release
   filter { "platforms:x64", "configurations:Release" }
      targetsuffix "64"
      defines      { "NDEBUG" }
      optimize     "Full"

   filter {}

   -- Copy contrib binaries (Windows only)
   copydir("./contrib/bin/win64", "./dist/bin/Debug/")
   copydir("./contrib/bin/win64", "./dist/bin/Release/")

   -- Bake kernels if requested
   if _OPTIONS["bakeKernel"] then
      defines { "ORO_PP_LOAD_FROM_STRING" }
      if os.ishost("windows") then
         os.execute(".\\tools\\bakeKernel.bat")
      else
         os.execute("./tools/bakeKernel.sh")
      end
   end

   -- Precompiled kernels
   if _OPTIONS["precompiled"] then
      defines { "ORO_PRECOMPILED" }
   end

   -- CUDA support (auto-detected or forced)
   include "./Orochi/enable_cuew"

-- -----------------------------------------------------------------------------
-- Projects
-- -----------------------------------------------------------------------------

   include "./UnitTest"

   group "Demos"
      include "./Test"
      include "./Test/DeviceEnum"
      include "./Test/WMMA"
      include "./Test/Texture"

      if os.istarget("windows") then
         include "./Test/VulkanComputeSimple"
         include "./Test/RadixSort"
         include "./Test/simpleD3D12"
      end
