project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { "../" }

    files { "**.h", "**.cpp" }
    files { "../contrib/cuew/**.h",  "../contrib/cuew/**.cpp" }
    files { "../contrib/hipew/**.h", "../contrib/hipew/**.cpp" }

    -- Silence vendored CUEW/HIPEW so --warning=extra targets only our sources.
    silenceVendoredWarnings("../contrib/**")
