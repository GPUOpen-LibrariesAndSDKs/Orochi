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

-- Values are premake's own `warnings` tokens; the API matches them
-- case-insensitively, so the lowercase spelling here needs no translation.
newoption {
    trigger     = "warning",
    value       = "LEVEL",
    description = "Compiler warning level: off, default, extra",
    allowed     = {
        { "off",     "Disable warnings" },
        { "default", "Default warnings" },
        { "extra",   "Extra warnings" }
    },
    default     = "default"
}

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

function linkWin32SystemLibs()
    filter "system:windows"
        links {
            "kernel32", "user32", "gdi32", "winspool", "comdlg32",
            "advapi32", "shell32", "ole32", "oleaut32", "uuid",
            "odbc32", "odbccp32"
        }
    filter {}
end

-- Adds a prebuild step running a helper script from `directory`. Branching on
-- the target system rather than the generation host keeps cross-generation
-- correct. On Windows plain `cd` does not change drive, hence `/d`.
function prebuildScript(directory, windowsCommand, posixCommand)
    filter "system:windows"
        prebuildcommands { 'cd /d "' .. directory .. '" && ' .. windowsCommand }
    filter "system:not windows"
        prebuildcommands { 'cd "' .. directory .. '" && ' .. posixCommand }
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

    -- A StaticLib is assembled by `ar`, so a linker choice there is dead weight.
    if _OPTIONS["clang"] then
        toolset "clang"
        filter "kind:not StaticLib"
            linker "LLD"
        filter {}
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

    externalwarnings "Off"
    warnings (_OPTIONS["warning"])

    -- Pinned rather than left to the toolset default so a consumer linking
    -- Orochi cannot end up mixing /MD and /MT.
    staticruntime "Off"

    filter "system:windows"
        defines      { "__WINDOWS__", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
    filter { "system:windows", "toolset:msc-v*" }
        conformancemode         "On"
        usestandardpreprocessor "On"
    filter {}

    filter "options:bakeKernel"
        defines { "ORO_PP_LOAD_FROM_STRING" }
    filter "options:precompiled"
        defines { "ORO_PRECOMPILED" }
    filter {}

-- -----------------------------------------------------------------------------
-- Projects
-- -----------------------------------------------------------------------------

    include "./Orochi"
    include "./ParallelPrimitives"
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
