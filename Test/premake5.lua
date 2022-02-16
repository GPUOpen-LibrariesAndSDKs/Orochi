project "Test"
      kind "ConsoleApp"

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

      links{ "version" }

      includedirs { "../" }
      files { "../Orochi/**.h", "../Orochi/**.cpp" }
      files { "*.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }
