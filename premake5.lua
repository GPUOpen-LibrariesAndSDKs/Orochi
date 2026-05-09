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
    description = "Directory for generated build files (default: build)"
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

-- Copy files from src_dir to dst_dir with optional glob filter
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

-- -----------------------------------------------------------------------------
-- Workspace Definition
-- -----------------------------------------------------------------------------

workspace "YamatanoOrochi"
    configurations { "Debug", "RelWithDebInfo", "Release" }
    platforms      { "x64" }
    language       "C++"
    cppdialect     "C++20"
    location       (_OPTIONS["builddir"] or "build")
    targetdir      "dist/bin/%{cfg.buildcfg}"
    startproject   "Unittest"

    -- Warning level (default: on)
    warnings "Default"
    filter "options:warning=off"
        warnings "Off"
    filter "options:warning=extra"
        warnings "Extra"
    filter {}

    -- Toolset selection (clangcl is Windows-only; must use Lua if, not filter)
    filter "system:macosx"
        toolset "clang"
        linker  "LLD"
    filter {}

    if _OPTIONS["clang"] then
        toolset "clang"
        linker  "LLD"
    end

    -- Platform-specific settings
    filter "system:windows"
        defines      { "__WINDOWS__", "_WIN32", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
        buildoptions { "/wd4244", "/wd4305", "/wd4018" }
    filter "system:linux"
        links { "dl" }
    filter {}

    -- Common defines
    defines { "_CRT_SECURE_NO_WARNINGS" }

    -- Platform architecture
    filter "platforms:x64"
        architecture "amd64"
    filter {}

    -- Build-wide compiler settings
    multiprocessorcompile "On"
    pic "On"

    -- Configuration: Debug
    filter { "platforms:x64", "configurations:Debug" }
        targetsuffix "64D"
        defines      { "DEBUG", "_DEBUG", "BUILD_CONFIG=\"Debug\"" }
        symbols      "On"
        optimize     "Off"
        runtime      "Debug"
    -- Configuration: RelWithDebInfo
    filter { "platforms:x64", "configurations:RelWithDebInfo" }
        targetsuffix "64"
        defines      { "NDEBUG", "BUILD_CONFIG=\"RelWithDebInfo\"" }
        symbols      "On"
        editandcontinue "Off"
        optimize     "Debug"
        linktimeoptimization "Fast"
        runtime      "Release"
    -- Configuration: Release
    filter { "platforms:x64", "configurations:Release" }
        targetsuffix "64"
        defines      { "NDEBUG", "BUILD_CONFIG=\"Release\"" }
        optimize     "Full"
        linktimeoptimization "Fast"
        runtime      "Release"
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
    copydir("./contrib/bin/win64", "./dist/bin/Debug/")
    copydir("./contrib/bin/win64", "./dist/bin/RelWithDebInfo/")
    copydir("./contrib/bin/win64", "./dist/bin/Release/")

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
