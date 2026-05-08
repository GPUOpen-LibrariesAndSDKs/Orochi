project "VulkanComputeSimple"
    kind "ConsoleApp"

    location "%{wks.location}/%{prj.name}"

    filter "system:windows"
        buildoptions { "/wd4244" }
        links {
            "kernel32", "user32", "gdi32", "winspool", "comdlg32",
            "advapi32", "shell32", "ole32", "oleaut32", "uuid",
            "odbc32", "odbccp32", "version"
        }
    filter {}

    includedirs { "../../" }
    includedirs { "./" }
    files { "../../Orochi/Orochi.h", "../../Orochi/Orochi.cpp" }
    files { "*.cpp" }
    files { "../../contrib/**.h", "../../contrib/**.cpp" }
