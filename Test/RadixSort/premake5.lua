project "RadixSort"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    if os.istarget("windows") then
        links { "version" }
    end

    includedirs { "../../" }
    files { "../../Orochi/**.h", "../../Orochi/**.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
    files { "*.cpp" }
    files { "../../ParallelPrimitives/**.h", "../../ParallelPrimitives/**.cpp" }
