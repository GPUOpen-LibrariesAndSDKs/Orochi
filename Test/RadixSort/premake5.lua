project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()

    files { "*.cpp" }
    files { "../../ParallelPrimitives/**.h", "../../ParallelPrimitives/**.cpp" }
