-- =============================================================================
-- Orochi core — static library
-- =============================================================================

project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { "../" }

    files { "../Orochi/**.h", "../Orochi/**.cpp" }
    files { "../contrib/cuew/**.h",  "../contrib/cuew/**.cpp" }
    files { "../contrib/hipew/**.h", "../contrib/hipew/**.cpp" }
