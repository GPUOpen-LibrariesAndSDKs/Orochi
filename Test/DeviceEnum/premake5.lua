project "DeviceEnum"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    linkVersionLib()

    includedirs { "../../" }
    files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "*.cpp" }
