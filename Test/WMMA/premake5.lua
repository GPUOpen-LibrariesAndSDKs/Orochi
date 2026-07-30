project "WMMA"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "*.h", "*.cpp" }
    files { "half.hpp" }
