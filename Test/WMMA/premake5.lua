project "WMMA"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "*.h", "*.cpp" }
    files { "contrib/**.hpp" }

    -- -isystem keeps vendored half.hpp out of the warning set.
    externalincludedirs { "contrib" }
