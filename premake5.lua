-- Orochi (YamatanoOrochi) workspace definition, build options, and helpers.

-- -----------------------------------------------------------------------------
-- Command-line Options
-- -----------------------------------------------------------------------------

newoption {
    trigger     = "clang",
    description = "Use Clang toolset instead of the default (GCC on Linux, MSVC on Windows)"
}

newoption {
    trigger     = "bakeKernel",
    description = "Bake GPU kernels into source as string literals"
}

newoption {
    trigger     = "precompiled",
    description = "Use precompiled kernels"
}

newoption {
    trigger     = "kernelcompile",
    description = "Compile kernels used for unit test"
}

newoption {
    trigger     = "builddir",
    value       = "PATH",
    description = "Directory for generated build files (default: .)"
}

newoption {
    trigger     = "warning",
    value       = "LEVEL",
    description = "Compiler warning level: off, on, extra (default: on)",
    allowed     = {
        { "off",   "Disable warnings" },
        { "on",    "Default warnings" },
        { "extra", "Extra warnings" }
    },
    default     = "on"
}

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

local rootDir = _MAIN_SCRIPT_DIR

function linkWin32SystemLibs()
    filter "system:windows"
        links {
            "kernel32", "user32", "gdi32", "winspool", "comdlg32",
            "advapi32", "shell32", "ole32", "oleaut32", "uuid",
            "odbc32", "odbccp32"
        }
    filter {}
end

-- Run a helper script, aborting generation if it fails.
function runScript(command)
    local ok, _, code = os.execute(command)
    if not ok then
        error("command failed (exit " .. tostring(code) .. "): " .. command)
    end
end

-- Run a helper script from a given directory. On Windows plain `cd` does not
-- change drive, so `/d` is required when the repository lives off C:.
function runScriptIn(directory, command)
    if os.ishost("windows") then
        runScript('cd /d "' .. directory .. '" && ' .. command)
    else
        runScript('cd "' .. directory .. '" && ' .. command)
    end
end

-- Windows has no rpath, so runtime DLLs are staged next to the binaries.
-- Called from a single project because every configuration shares one targetdir.
function stageWindowsRuntimeDlls()
    local contribBinDir = path.join(rootDir, "contrib/bin/win64")
    if not os.isdir(contribBinDir) then
        return
    end
    filter "system:windows"
        postbuildcommands { "{COPYDIR} %[" .. contribBinDir .. "] \"%{cfg.targetdir}\"" }
    filter {}
end

-- -----------------------------------------------------------------------------
-- Workspace Definition
-- -----------------------------------------------------------------------------

local buildConfigs = { "Debug", "DebugFast", "RelWithDebInfo", "Release" }

workspace "YamatanoOrochi"
    configurations (buildConfigs)
    platforms      { "x64" }
    language       "C++"
    cppdialect     "C++20"
    architecture   "amd64"
    location       (_OPTIONS["builddir"] or ".")
    targetdir      "dist/bin/%{cfg.buildcfg}"
    startproject   "UnitTest"

    multiprocessorcompile "On"
    systemversion "latest"

    filter "kind:StaticLib or SharedLib"
        pic "On"
    filter {}

    -- LLD's Mach-O support is deprecated, so keep the macOS default linker.
    filter "system:macosx"
        toolset "clang"
    filter {}

    if _OPTIONS["clang"] then
        toolset "clang"
        linker  "LLD"
    end

    filter "configurations:Debug or DebugFast"
        targetsuffix "64D"
    filter "configurations:RelWithDebInfo or Release"
        targetsuffix "64"
    filter {}

    filter "configurations:Debug"
        defines      { "DEBUG", "_DEBUG" }
        symbols      "Full"
        optimize     "Off"
        runtime      "Debug"
        editandcontinue "On"
    filter "configurations:DebugFast"
        defines      { "DEBUG", "_DEBUG", "_DEBUGFAST" }
        symbols      "Full"
        optimize     "Debug"
        runtime      "Debug"
        editandcontinue "On"
    filter "configurations:RelWithDebInfo"
        defines      { "NDEBUG" }
        symbols      "On"
        optimize     "On"
        runtime      "Release"
        intrinsics   "On"
        editandcontinue "Off"
    filter "configurations:Release"
        defines      { "NDEBUG" }
        symbols      "Off"
        optimize     "Full"
        runtime      "Release"
        intrinsics   "On"
        editandcontinue "Off"
    -- LTO is skipped for StaticLib: an IR-only libOrochi cannot be consumed by
    -- projects that link without LTO, and Orochi ships as a static library.
    filter { "configurations:Release", "kind:not StaticLib" }
        linktimeoptimization "Fast"
    filter { "configurations:Release", "toolset:msc-v*" }
        buildoptions { "/favor:AMD64" }
    filter {}

    local warningLevels = { off = "Off", on = "Default", extra = "Extra" }
    externalwarnings "Off"
    warnings (warningLevels[_OPTIONS["warning"]] or "Default")

    filter "system:windows"
        defines      { "__WINDOWS__", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
    filter { "system:windows", "toolset:msc-v*" }
        buildoptions { "/wd4244", "/wd4305", "/wd4018" }
    filter {}

    filter "options:bakeKernel"
        defines { "ORO_PP_LOAD_FROM_STRING" }
    filter "options:precompiled"
        defines { "ORO_PRECOMPILED" }
    filter {}

    -- The bake scripts write to paths relative to the repository root, which is
    -- premake's working directory.
    if _OPTIONS["bakeKernel"] then
        if os.ishost("windows") then
            runScript('"' .. path.join(rootDir, "tools/bakeKernel.bat") .. '"')
        else
            runScript('sh "' .. path.join(rootDir, "tools/bakeKernel.sh") .. '"')
        end
    end

-- -----------------------------------------------------------------------------
-- Projects
-- -----------------------------------------------------------------------------

    include "./Orochi"
    include "./UnitTest"

    group "Demos"
        include "./Test"
        include "./Test/DeviceEnum"
        include "./Test/WMMA"
        include "./Test/Texture"

        if os.istarget("windows") then
            include "./Test/VulkanComputeSimple"
            include "./Test/RadixSort"
            include "./Test/SimpleD3D12"
        end
