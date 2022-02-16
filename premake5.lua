workspace "YamatanoOrochi"
   configurations { "Debug", "Release" }
   language "C++"
   platforms "x64"
   architecture "x86_64"
   cppdialect "C++17"

   if os.istarget("windows") then
     defines{ "__WINDOWS__" }
     characterset ("MBCS")
     defines{ "_WIN32" }
   end
   if os.istarget("macosx") then
      buildToolset = "clang"
   end
   if os.istarget("linux") then
      buildoptions{ "-ldl" }
   end

  filter {"platforms:x64", "configurations:Debug"}
     targetsuffix "64D"
     defines { "DEBUG" }
     symbols "On"

  filter {"platforms:x64", "configurations:Release"}
     targetsuffix "64"
     defines { "NDEBUG" }
     optimize "On"
   filter {}
   if os.istarget("windows") then
      buildoptions { "/wd4244", "/wd4305", "/wd4018", "/wd4244" }
   end
   buildoptions{ "-Wno-ignored-attributes" }
   startproject "Test"

   include "./Test"
   include "./Test/DeviceEnum"
   group "Advanced"
      include "./Test/VulkanComputeSimple"
