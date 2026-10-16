project "Texture"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    uses { "Orochi" }

    links { "stb" }

    files { "texture_test_kernel.hpp", "*.cpp" }
