-- Orochi static library. Self-contained so external projects can `include` this
-- directory directly without pulling in the Orochi workspace.

-- Repository root, resolved from this script's own location rather than the
-- including workspace, which may live anywhere.
local orochiRoot = path.getabsolute("..", _SCRIPT_DIR)

-- Runs the CUDA SDK detection once; orochiApplyCuew() is callable afterwards.
include(path.join(orochiRoot, "Orochi/enable_cuew"))

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

-- Windows has no rpath, so the HIP runtime DLLs are staged next to the
-- binaries. Attached to Orochi because every executable links it, which makes
-- the copy happen regardless of which projects the workspace builds.
local function stageWindowsRuntimeDlls()
    local contribBinDir = path.join(orochiRoot, "contrib/bin/win64")
    if not os.isdir(contribBinDir) then
        return
    end
    filter "system:windows"
        postbuildcommands { "{COPYDIR} %[" .. contribBinDir .. "] \"%{cfg.targetdir}\"" }
    filter {}
end

-- Call from a consuming project to compile and link against Orochi.
-- The wranglers are named explicitly because premake does not propagate a
-- static library's own links to its consumers; order matters for GNU ld.
function useOrochi()
    externalincludedirs { orochiRoot }
    links { "Orochi", "cuew", "hipew" }
    orochiPlatformSettings()
end

-- Vendored wranglers live in their own projects so `warnings "Off"` applies at
-- project scope: premake's per-file `warnings` is honoured only by the Visual
-- Studio exporter, so a file filter would leave GCC/Clang unsilenced.
project "cuew"
    kind "StaticLib"
    location "%{wks.location}/contrib/cuew"
    warnings "Off"
    includedirs { orochiRoot }
    files { path.join(orochiRoot, "contrib/cuew/**.h"), path.join(orochiRoot, "contrib/cuew/**.cpp") }

project "hipew"
    kind "StaticLib"
    location "%{wks.location}/contrib/hipew"
    warnings "Off"
    includedirs { orochiRoot }
    files { path.join(orochiRoot, "contrib/hipew/**.h"), path.join(orochiRoot, "contrib/hipew/**.cpp") }

project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { orochiRoot }

    files { path.join(orochiRoot, "Orochi/**.h"), path.join(orochiRoot, "Orochi/**.cpp") }

    links { "cuew", "hipew" }

    orochiPlatformSettings()
    stageWindowsRuntimeDlls()
