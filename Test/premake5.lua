project "SimpleDemo"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    uses { "Orochi" }

    files { "*.cpp" }
