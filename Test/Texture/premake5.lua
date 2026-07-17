project "Texture"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "../../UnitTest/contrib/stb/**.h", "../../UnitTest/contrib/stb/**.cpp" }
    files { "texture_test_kernel.hpp", "*.cpp" }
