

--
-- In order to have Orochi compiled with CUDA, you need to define OROCHI_ENABLE_CUEW, and add the CUDA include path to your Orochi project.
--
-- If your project is using premake, this script can be included:
-- it helps to configure your Orochi project
--
--
--


function joinPaths_(basePath, additionalPath)
	-- Detect the path separator based on the operating system
	local pathSeparator = package.config:sub(1,1)
	-- Check if the basePath already ends with a path separator
	if basePath:sub(-1) ~= pathSeparator then
		basePath = basePath .. pathSeparator
	end
	return basePath .. additionalPath
end



-- search if CUDA PATH is inside a classic env var
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
	includedirs {  joinPaths_(cuda_path,"include") }
end


