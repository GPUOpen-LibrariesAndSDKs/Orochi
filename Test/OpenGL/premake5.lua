project "OpenGL"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

   if os.istarget("windows") then
      links{ "version" }
      libdirs{ "../../contrib/glew", "../../contrib/glfw/" }
      links{ "glew32s", "glfw3", "opengl32" }
   end

      includedirs { "../../" }
      files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
      files { "*.cpp" }
