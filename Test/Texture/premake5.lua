project "Texture"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "../../UnitTest/contrib/stb/**.h", "../../UnitTest/contrib/stb/**.cpp" }
    files { "texture_test_kernel.hpp", "*.cpp" }

    -- Silence vendored stb so --warning=extra targets only our sources.
    filter "files:../../UnitTest/contrib/**"
        warnings "Off"
    filter {}
