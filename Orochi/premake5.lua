-- Orochi static library; external workspaces `include` this directory and `uses { "Orochi" }`.

local orochiRoot = path.getabsolute("..", _SCRIPT_DIR)

-- In-repo consumers need the root as a plain include dir, or -isystem hides first-party warnings.
local isOrochiWorkspace = path.getabsolute(_MAIN_SCRIPT_DIR) == orochiRoot

include(path.join(orochiRoot, "Orochi/enable_cuew"))

-- The cd stays scoped: exporters append build-dir-relative commands (Ninja's stamp touch).
-- Ends with `filter {}`, so call it at project scope.
function orochiPrebuildScript(directory, windowsCommand, posixCommand)
    filter "system:windows"
        prebuildcommands { 'pushd "' .. directory .. '" && ' .. windowsCommand .. ' && popd' }
    filter "system:not windows"
        prebuildcommands { '( cd "' .. directory .. '" && ' .. posixCommand .. ' )' }
    filter {}
end

-- Windows has no rpath, so the HIP runtime DLLs are copied next to the current project's binaries
-- (xcopy /D skips up-to-date files). Public for host workspaces that stage them for their own
-- applications. Ends with `filter {}`, so call it at project scope. The trailing "/" makes premake
-- append the file name; a bare directory becomes xcopy's "dir*", which names a file.
function orochiStageRuntimeDlls()
    filter "system:windows"
        for _, dll in ipairs(os.matchfiles(path.join(orochiRoot, "contrib/bin/win64/*.dll"))) do
            postbuildcommands { "{COPYFILEIFNEWER} %[" .. dll .. "] %[%{!cfg.targetdir}/]" }
        end
    filter {}
end

-- Separate projects because per-file `warnings "Off"` only works in Visual Studio.
-- The `thirdparty` tag turns warnings off in the workspace's tags filter.
project "cuew"
    kind "StaticLib"
    location "%{wks.location}/contrib/cuew"
    tags { "thirdparty" }
    includedirs { orochiRoot }
    files { path.join(orochiRoot, "contrib/cuew/**.h"), path.join(orochiRoot, "contrib/cuew/**.cpp") }
    orochiApplyCuew()

project "hipew"
    kind "StaticLib"
    location "%{wks.location}/contrib/hipew"
    tags { "thirdparty" }
    includedirs { orochiRoot }
    files { path.join(orochiRoot, "contrib/hipew/**.h"), path.join(orochiRoot, "contrib/hipew/**.cpp") }

project "Orochi"
    kind "StaticLib"

    location "%{wks.location}/Orochi"

    includedirs { orochiRoot }

    files { path.join(orochiRoot, "Orochi/**.h"), path.join(orochiRoot, "Orochi/**.cpp") }

    links { "cuew", "hipew" }

    -- Here the binaries share this library's targetdir. A host workspace keeps its libraries
    -- elsewhere and calls orochiStageRuntimeDlls() from its own applications instead.
    if isOrochiWorkspace then
        orochiStageRuntimeDlls()
    end

    usage "PUBLIC"
        orochiApplyCuew()

    -- A usage does not link its own project, so list the archives in GNU ld order.
    usage "INTERFACE"
        if isOrochiWorkspace then
            includedirs { orochiRoot }
        else
            externalincludedirs { orochiRoot }
        end
        links { "Orochi", "cuew", "hipew" }
        filter "system:linux"
            links { "dl" }
        filter "system:windows"
            links { "version" }
        filter {}
