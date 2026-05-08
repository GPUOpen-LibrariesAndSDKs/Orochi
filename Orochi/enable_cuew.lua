-- =============================================================================
-- Orochi CUDA/CUEW Configuration
-- =============================================================================
-- Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.
-- Include this file from the root premake5.lua to auto-detect CUDA.

-- -----------------------------------------------------------------------------
-- Utility functions
-- -----------------------------------------------------------------------------

local function joinPaths(basePath, additionalPath)
   local sep = package.config:sub(1, 1)
   if basePath:sub(-1) ~= sep then
      basePath = basePath .. sep
   end
   return basePath .. additionalPath
end

local function isValidPath(p)
   return p ~= nil and p ~= "" and os.isdir(p)
end

-- -----------------------------------------------------------------------------
-- CUDA path detection
-- -----------------------------------------------------------------------------

-- Preferred CUDA version for this Orochi build
local cudaVersionName    = "12.2"
local cudaEnvVar         = "CUDA_PATH_V12_2"
local cudaPathLinux      = "/usr/local/cuda-12.2"
local cudaPathWindows    = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2"

-- Fallback paths
local backupCudaEnvVar   = "CUDA_PATH"
local backupCudaLinux    = "/usr/local/cuda"

-- Resolve CUDA SDK path (preferred version first, then fallback)
local cuda_path = os.getenv(cudaEnvVar)

if not isValidPath(cuda_path) and os.isdir(cudaPathLinux) then
   cuda_path = cudaPathLinux
end

if not isValidPath(cuda_path) and os.isdir(cudaPathWindows) then
   cuda_path = cudaPathWindows
end

if not isValidPath(cuda_path) then
   print("The required version of CUDA for this Orochi is not found: " .. cudaVersionName .. ". It's advised that you install this version.")
end

-- Try fallback paths
if not isValidPath(cuda_path) then
   cuda_path = os.getenv(backupCudaEnvVar)
end

if not isValidPath(cuda_path) and os.isdir(backupCudaLinux) then
   cuda_path = backupCudaLinux
end

-- -----------------------------------------------------------------------------
-- Apply CUDA configuration
-- -----------------------------------------------------------------------------

if _OPTIONS["forceCuda"] or isValidPath(cuda_path) then
   print("CUEW is enabled.")
   defines { "OROCHI_ENABLE_CUEW" }
end

if not isValidPath(cuda_path) then
   if _OPTIONS["forceCuda"] then
      print("WARNING: CUEW is enabled but it may not compile because CUDA SDK folder (CUDA_PATH) not found. You should install the CUDA SDK, or set CUDA_PATH.")
   else
      print("WARNING: CUEW is automatically disabled because CUDA SDK folder (CUDA_PATH) not found. You can force CUEW with the --forceCuda argument.")
   end
else
   print("CUDA SDK install folder found: " .. cuda_path)
   includedirs { joinPaths(cuda_path, "include") }
end
