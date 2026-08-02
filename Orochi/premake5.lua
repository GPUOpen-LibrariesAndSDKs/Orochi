-- Orochi static library. Self-contained so external projects can `include` this
-- directory directly without pulling in the Orochi workspace.

-- Repository root, resolved from this script's own location rather than the
-- including workspace, which may live anywhere.
local orochiRoot = path.getabsolute("..", _SCRIPT_DIR)

-- Runs the CUDA SDK detection once; orochiApplyCuew() is callable afterwards.
include(path.join(orochiRoot, "Orochi/enable_cuew"))

-- Kept local rather than reusing the workspace helper so this script stays
-- usable on its own. The Ninja generator ignores per-file `warnings`, so the
-- flag is also passed explicitly; MSVC is left to `warnings` alone because /w
-- on top of /W* emits an override diagnostic.
local function silenceVendoredWarnings(pattern)
    filter { "files:" .. pattern }
        warnings "Off"
    filter { "files:" .. pattern, "not toolset:msc-v*" }
        buildoptions { "-w" }
    filter {}
end

-- Applied to Orochi itself and re-applied to every consumer by useOrochi(),
-- so both sides agree on OROCHI_ENABLE_CUEW and the CUDA include path.
local function orochiPlatformSettings()
    filter "system:linux"
        links { "dl" }
    filter "system:windows"
        links { "version" }
    filter {}

    orochiApplyCuew()
end

-- Call from a consuming project to compile and link against Orochi.
function useOrochi()
    externalincludedirs { orochiRoot }
    links { "Orochi" }
    orochiPlatformSettings()
end

project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { orochiRoot }

    files { path.join(orochiRoot, "Orochi/**.h"), path.join(orochiRoot, "Orochi/**.cpp") }
    files { path.join(orochiRoot, "contrib/cuew/**.h"),  path.join(orochiRoot, "contrib/cuew/**.cpp") }
    files { path.join(orochiRoot, "contrib/hipew/**.h"), path.join(orochiRoot, "contrib/hipew/**.cpp") }

    orochiPlatformSettings()

    -- Silence vendored CUEW/HIPEW so --warning=extra targets only our sources.
    -- The pattern stays script-relative: `files:` filters match against paths
    -- resolved from this script, and an absolute pattern matches nothing.
    silenceVendoredWarnings("../contrib/**")
