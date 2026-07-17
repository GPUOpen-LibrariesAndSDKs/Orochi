-- =============================================================================
-- Orochi CUDA/CUEW Configuration
-- =============================================================================
-- Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.

-- -----------------------------------------------------------------------------
-- Utility functions
-- -----------------------------------------------------------------------------

local function isValidPath(p)
    return p ~= nil and p ~= "" and os.isdir(p)
end

-- -----------------------------------------------------------------------------
-- CUDA path detection
-- -----------------------------------------------------------------------------

-- Candidate CUDA versions, most preferred first.
local cudaVersions = { "12.2" }

-- Return the first existing SDK path for a given version.
local function findCudaVersion(version)
    local envVar = "CUDA_PATH_V" .. version:gsub("%.", "_")
    local candidates = {
        os.getenv(envVar),
        "/usr/local/cuda-" .. version,
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v" .. version,
    }
    for _, p in ipairs(candidates) do
        if isValidPath(p) then
            return p
        end
    end
    return nil
end

-- Resolve CUDA SDK path: preferred versions first, then fallbacks.
local cuda_path = nil
for _, version in ipairs(cudaVersions) do
    cuda_path = findCudaVersion(version)
    if isValidPath(cuda_path) then
        break
    end
end

if not isValidPath(cuda_path) then
    print("The required version of CUDA for this Orochi is not found: " .. table.concat(cudaVersions, ", ") .. ". It's advised that you install one of these versions.")
end

-- Try fallback paths
if not isValidPath(cuda_path) then
    cuda_path = os.getenv("CUDA_PATH")
end

if not isValidPath(cuda_path) and os.isdir("/usr/local/cuda") then
    cuda_path = "/usr/local/cuda"
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
    externalincludedirs { path.join(cuda_path, "include") }
end
