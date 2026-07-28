project "VulkanComputeSimple"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()
    linkWin32SystemLibs()
    includedirs { "./" }
    files { "*.cpp" }
