project "DeviceEnum"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

      buildoptions { "/wd4244" }

      links{ "version" }

      includedirs { "../../" }
      files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "*.cpp" }
