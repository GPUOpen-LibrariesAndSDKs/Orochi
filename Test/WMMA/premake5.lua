project "WMMA"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    linkVersionLib()

    includedirs { "../../" }
    files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "*.h", "*.cpp" }
    files { "half.hpp" }
