project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    uses { "ParallelPrimitives" }

    files { "*.cpp" }
