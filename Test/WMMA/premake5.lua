project "WMMA"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()

    files { "*.h", "*.cpp" }
    files { "half.hpp" }
