project "VulkanComputeSimple"
    kind "ConsoleApp"

    location "%{wks.location}/Test/%{prj.name}"

    filter "system:windows"
        links {
            "kernel32", "user32", "gdi32", "winspool", "comdlg32",
            "advapi32", "shell32", "ole32", "oleaut32", "uuid",
            "odbc32", "odbccp32"
        }
    filter {}

    useOrochi()
    linkVersionLib()
    includedirs { "./" }
    files { "*.cpp" }
