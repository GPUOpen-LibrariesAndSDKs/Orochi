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

-- -----------------------------------------------------------------------------
-- Utility Functions
-- -----------------------------------------------------------------------------

function copydir(src_dir, dst_dir, filter, single_dst_dir)
	if not os.isdir(src_dir) then
		print("copydir FAILED: " .. src_dir .. " is not an existing directory!" )
		return nil
	end
	filter = filter or "**"
	src_dir = src_dir .. "/"
--	print("copy '%s' to '%s'.", src_dir .. filter, dst_dir)
	dst_dir = dst_dir .. "/"
	local dir = path.rebase(".",path.getabsolute("."), src_dir) -- root dir, relative from src_dir

	os.chdir( src_dir ) -- change current directory to src_dir
		local matches = os.matchfiles(filter)
	os.chdir( dir ) -- change current directory back to root

	local counter = 0
	for k, v in ipairs(matches) do
		local target = iif(single_dst_dir, path.getname(v), v)
		--make sure, that directory exists or os.copyfile() fails
		os.mkdir( path.getdirectory(dst_dir .. target))
		if os.copyfile( src_dir .. v, dst_dir .. target) then
			counter = counter + 1
		end
	end

	if counter == #matches then
--		print("    %d files copied.", counter)
		return true
	else
--		print("    %d/%d files copied.", counter, #matches)
		return nil
	end
end

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
    startproject   "Unittest"

    filter "kind:StaticLib or SharedLib"
        pic "On"
    filter {}

    filter "system:linux"
        links { "dl" }
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
    filter "configurations:DebugFast"
        defines      { "DEBUG", "_DEBUG", "_DEBUGFAST" }
        symbols      "Full"
        optimize     "Debug"
        runtime      "Debug"
    filter "configurations:RelWithDebInfo"
        defines      { "NDEBUG" }
        symbols      "On"
        optimize     "On"
        runtime      "Release"
    filter "configurations:Release"
        defines      { "NDEBUG" }
        symbols      "Off"
        optimize     "Full"
        runtime      "Release"
    -- No LTO for StaticLib: consumers linking without LTO cannot read IR-only archives.
    filter { "configurations:Release", "kind:not StaticLib" }
        linktimeoptimization "Fast"
    filter { "configurations:Release", "toolset:msc-v*" }
        buildoptions { "/favor:AMD64" }
    filter {}

    externalwarnings "Off"
    warnings (_OPTIONS["warning"])

    -- Pinned so consumers linking Orochi cannot mix /MD and /MT.
    staticruntime "Off"

    filter "system:windows"
        defines      { "__WINDOWS__", "_CRT_SECURE_NO_WARNINGS" }
        characterset "MBCS"
        multiprocessorcompile "On"
        systemversion "latest"
    filter { "system:windows", "configurations:Debug or DebugFast" }
        editandcontinue "On"
    filter { "system:windows", "configurations:RelWithDebInfo or Release" }
        intrinsics      "On"
        editandcontinue "Off"
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

    copydir("./contrib/bin/win64", "./dist/bin/Debug/")
    copydir("./contrib/bin/win64", "./dist/bin/RelWithDebInfo/")
    copydir("./contrib/bin/win64", "./dist/bin/Release/")
    if _OPTIONS["bakeKernel"] then
        if os.ishost("windows") then
            os.execute(".\\tools\\bakeKernel.bat")
        else
            os.execute(".\\tools\\bakeKernel.sh")
        end
    end

    -- try to enable CUDA if possible.
    include "./Orochi/enable_cuew"

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
