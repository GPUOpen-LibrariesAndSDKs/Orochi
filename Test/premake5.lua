project "Test"
      kind "ConsoleApp"

      targetdir "../dist/bin/%{cfg.buildcfg}"
      location "../build/"

   if os.istarget("windows") then
      links{ "version" }
   end
      includedirs { "../" }
      files { "../Orochi/**.h", "../Orochi/**.cpp" }
      files { "*.cpp" }
      files { "../contrib/**.h", "../contrib/**.cpp" }
