project "SimpleDemo"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()

    files { "*.cpp" }
