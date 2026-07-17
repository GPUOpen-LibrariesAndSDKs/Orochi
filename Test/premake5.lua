project "SimpleDemo"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    useOrochi()
    linkVersionLib()

    files { "*.cpp" }
