project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    linkVersionLib()

    includedirs { "../../" }
    files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "*.cpp" }
    files { "../../ParallelPrimitives/**.h", "../../ParallelPrimitives/**.cpp" }
