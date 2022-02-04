project "VulkanComputeSimple"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

      buildoptions { "/wd4244" }

--      links{ "Pop" }
      links{ "kernel32", "user32", "gdi32", "winspool", "comdlg32", "advapi32", "shell32", "ole32", "oleaut32", "uuid", "odbc32", "odbccp32", "version" }

      includedirs { "../../" }
      includedirs { "./" }
      files { "../../Pop/**.h", "../../Pop/**.cpp" }
      files { "*.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
