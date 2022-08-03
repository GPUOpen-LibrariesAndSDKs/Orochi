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

   include "./UnitTest"
   group "Samples"
   	include "./Test"
   	include "./Test/DeviceEnum"
   
     if os.istarget("windows") then
        group "Advanced"
        include "./Test/VulkanComputeSimple"
     end
