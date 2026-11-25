project "VulkanComputeSimple"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    uses { "Orochi" }
    linkWin32SystemLibs()
    includedirs { "./" }
    files { "*.cpp" }
