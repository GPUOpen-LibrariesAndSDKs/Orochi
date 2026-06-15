-- =============================================================================
-- Orochi (YamatanoOrochi) — Premake5 Build Configuration
-- =============================================================================

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
    trigger     = "forceCuda",
    description = "Force CUDA backend even if CUDA_PATH is not found (may cause compilation errors)"
}

newoption {
    trigger     = "builddir",
    value       = "PATH",
    description = "Directory for generated build files (default: project root)"
}

newoption {
    trigger     = "warning",
    value       = "LEVEL",
    description = "Compiler warning level: off, on, extra (default: on)",
    allowed     = {
        { "off",   "Disable warnings" },
        { "on",    "Default warnings" },
        { "extra", "Extra warnings" }
    }
}

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

-- Copy files matching `filter` (default "**") from `src_dir` into `dst_dir`.
-- With `single_dst_dir`, flatten the tree so every file lands directly in
-- `dst_dir`. Returns true on success, nil if any matched file failed to copy.
function copydir(src_dir, dst_dir, filter, single_dst_dir)
    if not os.isdir(src_dir) then
        print("copydir FAILED: " .. src_dir .. " is not an existing directory!")
        return nil
    end

    filter = filter or "**"
    src_dir = src_dir .. "/"
    dst_dir = dst_dir .. "/"

    local dir = path.rebase(".", path.getabsolute("."), src_dir)
    os.chdir(src_dir)
    local matches = os.matchfiles(filter)
    os.chdir(dir)

    if #matches == 0 then
        return true
    end

    local counter = 0
    for _, v in ipairs(matches) do
        local target = iif(single_dst_dir, path.getname(v), v)
        os.mkdir(path.getdirectory(dst_dir .. target))
        if os.copyfile(src_dir .. v, dst_dir .. target) then
            counter = counter + 1
        end
    end

    return counter == #matches or nil
end

-- Link the Windows "version" library, required by every Orochi test target.
-- Wrapped so the per-project filter block is not copy-pasted into each one.
function linkVersionLib()
    filter "system:windows"
        links { "version" }
    filter {}
end

-- -----------------------------------------------------------------------------
-- Workspace Definition
-- -----------------------------------------------------------------------------

local buildConfigs = { "Debug", "RelWithDebInfo", "Release" }

workspace "YamatanoOrochi"
    configurations (buildConfigs)
    platforms      { "x64" }
    language       "C++"
    cppdialect     "C++20"
    location       (_OPTIONS["builddir"] or ".")
    targetdir      "dist/bin/%{cfg.buildcfg}"
    startproject   "UnitTest"

    -- Build-wide compiler settings
    multiprocessorcompile "On"
    pic "On"

    -- Toolset selection. macOS always uses clang; --clang opts in elsewhere.
    filter "system:macosx"
        toolset "clang"
        linker  "LLD"
    filter {}

    if _OPTIONS["clang"] then
        toolset "clang"
        linker  "LLD"
    end

    -- Platform architecture
    filter "platforms:x64"
        architecture "amd64"
    filter {}

    -- Per-config target name suffix
    filter { "platforms:x64", "configurations:Debug" }
        targetsuffix "64D"
    filter { "platforms:x64", "configurations:RelWithDebInfo or Release" }
        targetsuffix "64"
    filter {}

    -- Configuration: Debug
    filter { "configurations:Debug" }
        defines      { "DEBUG", "_DEBUG", "BUILD_CONFIG=\"Debug\"" }
        symbols      "Full"
        optimize     "Off"
        runtime      "Debug"
        floatingpointexceptions "On"
        editandcontinue "On"
    -- Configuration: RelWithDebInfo
    filter { "configurations:RelWithDebInfo" }
        defines      { "NDEBUG", "BUILD_CONFIG=\"RelWithDebInfo\"" }
        symbols      "On"
        optimize     "Debug"
        runtime      "Debug"
        floatingpointexceptions "On"
        editandcontinue "On"
    -- Configuration: Release
    filter { "configurations:Release" }
        defines      { "NDEBUG", "BUILD_CONFIG=\"Release\"" }
        symbols      "Off"
        optimize     "Full"
        runtime      "Release"
        intrinsics   "On"
        floatingpointexceptions "Off"
        editandcontinue "Off"
        linktimeoptimization "Fast"
    filter { "configurations:Release", "toolset:msc-v*" }
        buildoptions { "/favor:AMD64" }
    filter {}

    -- Warning level (default: on)
    local warningLevels = { off = "Off", on = "Default", extra = "Extra" }
    externalwarnings "Off"
    warnings (warningLevels[_OPTIONS["warning"]] or "Default")

    -- Platform-specific settings
    filter "system:windows"
        defines      { "__WINDOWS__", "_WIN32", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
        buildoptions { "/wd4244", "/wd4305", "/wd4018" }
    filter "system:linux"
        links { "dl" }
    filter {}

    -- Bake kernels if requested (os.execute runs at script time, not build time)
    filter "options:bakeKernel"
        defines { "ORO_PP_LOAD_FROM_STRING" }
    filter {}

    if _OPTIONS["bakeKernel"] then
        if os.ishost("windows") then
            os.execute("%[./tools/bakeKernel.bat]")
        else
            os.execute("%[./tools/bakeKernel.sh]")
        end
    end

    -- Precompiled kernels
    filter "options:precompiled"
        defines { "ORO_PRECOMPILED" }
    filter {}

    -- Copy contrib binaries (Windows only)
    if os.istarget("windows") then
        for _, cfg in ipairs(buildConfigs) do
            copydir("./contrib/bin/win64", "./dist/bin/" .. cfg .. "/")
        end
    end

    -- CUDA support (auto-detected or forced)
    include "./Orochi/enable_cuew"

-- -----------------------------------------------------------------------------
-- Projects
-- -----------------------------------------------------------------------------

    include "./UnitTest"

    group "Demos"
        include "./Test"
        include "./Test/DeviceEnum"
        include "./Test/WMMA"
        include "./Test/Texture"

        if os.istarget("windows") then
            include "./Test/VulkanComputeSimple"
            include "./Test/RadixSort"
            include "./Test/simpleD3D12"
        end
