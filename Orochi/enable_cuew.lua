-- Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.
--
-- Including this file applies the settings to the current scope, as before.
-- Because premake runs a given `include` only once, the settings are also
-- exposed as orochiApplyCuew() so every project that needs them can re-apply.

-- Declared here, not in the workspace, so external projects that include only
-- this file still accept --forceCuda. Guarded because a host workspace may
-- have registered the same trigger already.
if not premake.option.get("forceCuda") then
    newoption {
        trigger     = "forceCuda",
        description = "Force CUDA backend even if CUDA_PATH is not found (may cause compilation errors)"
    }
end

local function isValidPath(p)
    return p ~= nil and p ~= "" and os.isdir(p)
end

-- Most preferred first.
local cudaVersions = { "12.2" }

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

-- Preferred versions first, then CUDA_PATH, then the default install dir.
local cuda_path = nil
for _, version in ipairs(cudaVersions) do
    cuda_path = findCudaVersion(version)
    if isValidPath(cuda_path) then
        break
    end
end
local foundPreferredCudaVersion = isValidPath(cuda_path)

if not isValidPath(cuda_path) then
    cuda_path = os.getenv("CUDA_PATH")
end
if not isValidPath(cuda_path) and os.isdir("/usr/local/cuda") then
    cuda_path = "/usr/local/cuda"
end

-- Detection is reported once, at include time, rather than per project.
if isValidPath(cuda_path) then
    print("CUEW is enabled. CUDA SDK found: " .. cuda_path)
    if not foundPreferredCudaVersion then
        print("WARNING: preferred CUDA version not found (" .. table.concat(cudaVersions, ", ") .. "); using a fallback CUDA SDK install folder.")
    end
elseif _OPTIONS["forceCuda"] then
    print("WARNING: CUEW is force-enabled but CUDA SDK not found (set CUDA_PATH). Compilation may fail.")
else
    print("WARNING: CUEW disabled; CUDA SDK not found (preferred: " .. table.concat(cudaVersions, ", ") .. "). Use --forceCuda to override.")
end

-- Applies the detected CUDA settings to the current project scope.
function orochiApplyCuew()
    if isValidPath(cuda_path) then
        defines { "OROCHI_ENABLE_CUEW" }
        externalincludedirs { path.join(cuda_path, "include") }
    elseif _OPTIONS["forceCuda"] then
        defines { "OROCHI_ENABLE_CUEW" }
    end
end

orochiApplyCuew()
