project "WMMA"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    uses { "Orochi" }

    files { "*.h", "*.hpp", "*.cpp" }
