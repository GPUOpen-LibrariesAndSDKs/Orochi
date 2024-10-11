project "OpenGL"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

   if os.istarget("windows") then
      links{ "version" }
      libdirs{ "../../UnitTest/contrib/glew", "../../UnitTest/contrib/glfw/" }
      links{ "glew32s", "glfw3", "opengl32" }
   end

      includedirs { "../../" }
      files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "../../UnitTest/contrib/**.h", "../../UnitTest/contrib/**.cpp" }
      files { "*.cpp" }
