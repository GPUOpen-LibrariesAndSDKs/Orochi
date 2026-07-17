project "Texture"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "../../UnitTest/contrib/**.h", "../../UnitTest/contrib/**.cpp" }
    files { "texture_test_kernel.hpp", "*.cpp" }
