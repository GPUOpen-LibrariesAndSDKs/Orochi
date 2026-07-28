project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { "../" }

    files { "../Orochi/**.h", "../Orochi/**.cpp" }
    files { "../contrib/cuew/**.h",  "../contrib/cuew/**.cpp" }
    files { "../contrib/hipew/**.h", "../contrib/hipew/**.cpp" }

    -- Silence vendored CUEW/HIPEW so --warning=extra targets only our sources.
    filter "files:../contrib/**"
        warnings "Off"
    filter {}
