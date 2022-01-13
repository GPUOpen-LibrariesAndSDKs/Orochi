workspace "Pop"
   configurations { "Debug", "Release" }
   language "C++"
   platforms "x64"
   architecture "x86_64"

   defines{ "__WINDOWS__" }
   characterset ("MBCS")

  filter {"platforms:x64", "configurations:Debug"}
     targetsuffix "64D"
     defines { "DEBUG" }
     symbols "On"

  filter {"platforms:x64", "configurations:Release"}
     targetsuffix "64"
     defines { "NDEBUG" }
     optimize "On"
   filter {}
   buildoptions { "/wd4244", "/wd4305", "/wd4018" }

   defines{ "_WIN32" }

--   include "./Pop"
   include "./Test"