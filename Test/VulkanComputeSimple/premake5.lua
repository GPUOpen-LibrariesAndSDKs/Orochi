project "VulkanComputeSimple"
      kind "ConsoleApp"

      targetdir "../../dist/bin/%{cfg.buildcfg}"
      location "../../build/"

   if os.istarget("windows") then
      buildoptions { "/wd4244" }

--      links{ "Pop" }
      links{ "kernel32", "user32", "gdi32", "winspool", "comdlg32", "advapi32", "shell32", "ole32", "oleaut32", "uuid", "odbc32", "odbccp32", "version" }
   end

      includedirs { "../../" }

   -- The bundled headers (1.3.204) don't build on modern libstdc++, so Linux uses SDK/system ones.
   -- No Vulkan lib is linked anywhere: vulkan-hpp's RAII Context dlopens the loader at runtime.
   if os.istarget("windows") then
      includedirs { "./" }
   else
      local vulkanSdk = os.getenv("VULKAN_SDK")
      if vulkanSdk then
         includedirs { vulkanSdk .. "/include" }
      end
   end

      files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
      files { "*.cpp" }
      files { "../../contrib/**.h", "../../contrib/**.cpp" }
