project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    links { "ParallelPrimitives" }

    files { "*.cpp" }
