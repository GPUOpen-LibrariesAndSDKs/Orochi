project "Test"
      kind "ConsoleApp"

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

      buildoptions { "/wd4244" }

--      links{ "Pop" }
      links{ "version" }

      includedirs { "../" }
      files { "../Pop/**.h", "../Pop/**.cpp" }
      files { "*.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }
