project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    filter "system:windows"
        links { "version" }
    filter {}

    includedirs { "../../" }
    files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "*.cpp" }
    files { "../../ParallelPrimitives/**.h", "../../ParallelPrimitives/**.cpp" }
