project "Texture"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    linkVersionLib()

    includedirs { "../../" }
    files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "../../UnitTest/contrib/**.h", "../../UnitTest/contrib/**.cpp" }
    files { "texture_test_kernel.hpp", "*.cpp" }
