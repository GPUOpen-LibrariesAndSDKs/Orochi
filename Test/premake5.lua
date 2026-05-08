project "SimpleDemo"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    if os.istarget("windows") then
        links { "version" }
    end

    includedirs { "../" }
    files { "../Orochi/Orochi.h", "../Orochi/Orochi.cpp" }
    files { "*.cpp" }
    files { "../contrib/**.h", "../contrib/**.cpp" }
