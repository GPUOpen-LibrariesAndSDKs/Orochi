-- Orochi (YamatanoOrochi) workspace definition, build options, and helpers.

-- Snapshot before newoption so validateFlagOptions can tell premake's own options apart.
local builtinOptions = {}
for trigger in pairs(premake.option.list) do
    builtinOptions[trigger] = true
end

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
    trigger     = "distdir",
    value       = "PATH",
    description = "Directory for build products; binaries go to PATH/bin/<config> (default: dist)"
}

-- Values are premake `warnings` tokens, matched case-insensitively.
newoption {
    trigger     = "warning",
    value       = "LEVEL",
    description = "Compiler warning level: off, default, high, extra, everything",
    allowed     = {
        { "off",        "Disable warnings" },
        { "default",    "Default warnings" },
        { "high",       "High warnings" },
        { "extra",      "Extra warnings" },
        { "everything", "All warnings the compiler offers" }
    },
    default     = "default"
}

newoption {
    trigger     = "werror",
    description = "Treat all compiler warnings as errors"
}

newoption {
    trigger     = "sanitizer",
    value       = "MODE",
    description = "Enable a compiler sanitizer (mapped to premake's sanitize API)",
    allowed     = {
        { "none",    "Disable sanitizer support" },
        { "address", "Enable AddressSanitizer" },
        { "thread",  "Enable ThreadSanitizer (GCC and Clang only)" },
        { "undef",   "Enable UndefinedBehaviorSanitizer (GCC and Clang only)" }
    },
    default     = "none"
}

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

-- Ends with `filter {}`, so call it at project scope, never inside a filter.
function linkWin32SystemLibs()
    filter "system:windows"
        links {
            "kernel32", "user32", "gdi32", "winspool", "comdlg32",
            "advapi32", "shell32", "ole32", "oleaut32", "uuid",
            "odbc32", "odbccp32"
        }
    filter {}
end

-- A switch takes no value, so `--clang=foo` would otherwise be read as "on".
-- Built-in options are skipped because modules such as compilecommands declare values differently.
local function validateFlagOptions()
    for trigger, option in pairs(premake.option.list) do
        local given = _OPTIONS[trigger]
        if not builtinOptions[trigger] and option.value == nil and given ~= nil and given ~= "" then
            error(string.format("--%s does not take a value (got '%s').", trigger, given), 0)
        end
    end
end

validateFlagOptions()

local sanitizeModes = { address = "Address", thread = "Thread", undef = "UndefinedBehavior" }

-- -----------------------------------------------------------------------------
-- Workspace Definition
-- -----------------------------------------------------------------------------

local buildConfigs = { "Debug", "DebugFast", "RelWithDebInfo", "Release" }

workspace "YamatanoOrochi"
    configurations (buildConfigs)
    platforms      { "x64" }
    -- Otherwise premake's name sort makes DebugFast the default.
    defaultconfiguration "Debug"
    defaultplatform      "x64"
    language       "C++"
    cppdialect     "C++20"
    architecture   "x86_64"
    location       (_OPTIONS["builddir"] or ".")
    targetdir      (path.join(_OPTIONS["distdir"] or "dist", "bin/%{cfg.buildcfg}"))
    startproject   "UnitTest"

    -- `uses` can list a static library before its dependents; GNU ld needs a group.
    linkgroups "On"

    filter "kind:StaticLib or SharedLib"
        pic "On"
    filter {}

    multiprocessorcompile "On"

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
    filter "configurations:DebugFast"
        defines      { "DEBUG", "_DEBUG" }
        symbols      "Full"
        optimize     "Debug"
        runtime      "Debug"
    -- MSVC maps optimize "Debug" to /Od, so DebugFast would equal Debug. /RTC1 conflicts with /O2 (D8016).
    filter { "configurations:DebugFast", "toolset:msc-v*" }
        optimize     "Speed"
        buildoptions { "/Ob1" }
        runtimechecks "Off"
    filter "configurations:RelWithDebInfo"
        defines      { "NDEBUG" }
        symbols      "On"
        optimize     "On"
        runtime      "Release"
    filter "configurations:Release"
        defines      { "NDEBUG" }
        symbols      "Off"
        optimize     "Speed"
        runtime      "Release"
    filter {}

    -- Bounds checks that keep the libstdc++ ABI.
    filter { "system:linux", "configurations:Debug or DebugFast" }
        defines { "_GLIBCXX_ASSERTIONS" }
    filter "toolset:clang"
        buildoptions { "-fcolor-diagnostics" }
    filter {}

    -- https://premake.github.io/docs/sanitize/
    local sanitizeValue = sanitizeModes[_OPTIONS["sanitizer"]]
    if sanitizeValue then
        -- MSVC ships only AddressSanitizer and would silently emit no instrumentation.
        if os.istarget("windows") and not _OPTIONS["clang"] and sanitizeValue ~= "Address" then
            error("--sanitizer=" .. _OPTIONS["sanitizer"] .. " is not supported by MSVC. Use --sanitizer=address, or add --clang.", 0)
        end
        sanitize { sanitizeValue }
        omitframepointer "Off"
    end

    externalwarnings "Off"
    warnings (_OPTIONS["warning"])

    -- Pinned so consumers linking Orochi cannot mix /MD and /MT.
    staticruntime "Off"

    filter "system:windows"
        defines      { "__WINDOWS__", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
        systemversion "latest"
    -- /ZI cannot be combined with optimization, and MSVC rejects /INCREMENTAL alongside ASan.
    filter { "system:windows", "configurations:Debug" }
        editandcontinue (iif(sanitizeValue == nil, "On", "Off"))
    filter { "system:windows", "configurations:DebugFast or RelWithDebInfo or Release" }
        editandcontinue "Off"
    filter { "system:windows", "configurations:RelWithDebInfo or Release" }
        intrinsics      "On"
    filter { "system:windows", "toolset:msc-v*" }
        conformancemode         "On"
        usestandardpreprocessor "On"
    filter {}

    filter "options:werror"
        fatalwarnings { "All" }
    filter {}

    -- Must follow every warnings/fatalwarnings/conformance setting above so it overrides them.
    -- MSVC keeps /WX alongside /w, hence removefatalwarnings.
    filter "tags:thirdparty"
        warnings "Off"
        removefatalwarnings { "All" }
    filter { "tags:thirdparty", "toolset:msc-v*" }
        conformancemode         "Off"
        usestandardpreprocessor "Off"
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
