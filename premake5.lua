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
      links { "dl" }
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
   -- buildoptions{ "-Wno-ignored-attributes" }
   defines { "_CRT_SECURE_NO_WARNINGS" }
   startproject "Unittest"

   include "./UnitTest"
   group "Samples"
   	include "./Test"
   	include "./Test/DeviceEnum"
   
     if os.istarget("windows") then
        group "Advanced"
        include "./Test/VulkanComputeSimple"
     end
