project "SimpleDemo"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    filter "system:windows"
        links { "version" }
    filter {}

    includedirs { "../" }
    files { "../Orochi/Orochi.h", "../Orochi/Orochi.cpp" }
    files { "*.cpp" }
    files { "../contrib/**.h", "../contrib/**.cpp" }
