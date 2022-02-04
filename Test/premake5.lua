project "Test"
      kind "ConsoleApp"

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

      buildoptions { "/wd4244" }

--      links{ "Pop" }
      links{ "version" }

      includedirs { "../" }
      files { "../Orochi/**.h", "../Orochi/**.cpp" }
      files { "*.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }
