newoption {
    trigger = "bakeKernel",
    description = "bakeKernel"
}

newoption {
   trigger = "precompiled",
   description = "Use precompiled kernels"
}

newoption {
   trigger = "kernelcompile",
   description = "Compile kernels used for unit test"
}

newoption {
   trigger = "forceCuda",
   description = "By default, CUDA backend is enabled at compile-time only if the CUDA_PATH exists. Using this argument forces the activation of CUDA backend. However your project may have compilation errors."
}


function joinPaths(basePath, additionalPath)
	-- Detect the path separator based on the operating system
	local pathSeparator = package.config:sub(1,1)
	-- Check if the basePath already ends with a path separator
	if basePath:sub(-1) ~= pathSeparator then
		basePath = basePath .. pathSeparator
	end
	return basePath .. additionalPath
end


function copydir(src_dir, dst_dir, filter, single_dst_dir)
	if not os.isdir(src_dir) then
		printError("'%s' is not an existing directory!", src_dir)
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

workspace "YamatanoOrochi"
   configurations { "Debug", "Release" }
   language "C++"
   platforms "x64"
   architecture "x86_64"
   cppdialect "C++17"

   if os.istarget("windows") then
     defines{ "__WINDOWS__" }
     characterset ("MBCS")
     defines{ "_WIN32" }
   end
   if os.istarget("macosx") then
      buildToolset = "clang"
   end
   if os.istarget("linux") then
      links { "dl" }
   end

  filter {"platforms:x64", "configurations:Debug"}
     targetsuffix "64D"
     defines { "DEBUG" }
     symbols "On"

  filter {"platforms:x64", "configurations:Release"}
     targetsuffix "64"
     defines { "NDEBUG" }
     optimize "On"
   filter {}
   if os.istarget("windows") then
      buildoptions { "/wd4244", "/wd4305", "/wd4018", "/wd4244" }
   end
   -- buildoptions{ "-Wno-ignored-attributes" }
   defines { "_CRT_SECURE_NO_WARNINGS" }
   startproject "Unittest"

    copydir("./contrib/bin/win64", "./dist/bin/Debug/")
    copydir("./contrib/bin/win64", "./dist/bin/Release/")
	if _OPTIONS["bakeKernel"] then
		defines { "ORO_PP_LOAD_FROM_STRING" }
      if os.ishost("windows") then
		   os.execute(".\\tools\\bakeKernel.bat")
      else
         os.execute(".\\tools\\bakeKernel.sh")
      end
	end

   if _OPTIONS["precompiled"] then
		defines {"ORO_PRECOMPILED"}
	end



	-- search if CUDA PATH en inside a classic env var
	cuda_path = os.getenv("CUDA_PATH")

	-- if the CUDA PATH is not in the env var, search in the classic folder
	if (not os.istarget("windows")) and ( cuda_path == nil or cuda_path == '' ) then
		potentialUnixCudaPath = "/usr/local/cuda";
		if ( os.isdir(potentialUnixCudaPath) ) then
			cuda_path = potentialUnixCudaPath
		end
	end


	-- Enable CUEW if CUDA is forced or if we find the CUDA SDK folder
	if ( _OPTIONS["forceCuda"] or   ( cuda_path ~= nil and cuda_path ~= '' )  ) then
		print("CUEW is enabled.")
		defines {"OROCHI_ENABLE_CUEW"}
	end

	-- If we find the CUDA SDK folder, add it to the include dir
	if cuda_path == nil or cuda_path == '' then
		if _OPTIONS["forceCuda"] then
			print("WARNING: CUEW is enabled but it may not compile because CUDA SDK folder ( CUDA_PATH ) not found. You should install the CUDA SDK, or set CUDA_PATH.")
		else
			print("WARNING: CUEW is automatically disabled because CUDA SDK folder ( CUDA_PATH ) not found. You can force CUEW with the --forceCuda argument.")
		end
	else
		print("CUDA SDK install folder found: " .. cuda_path)
		includedirs {  joinPaths(cuda_path,"include") }
	end




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
