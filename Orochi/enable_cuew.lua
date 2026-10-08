-- Detects the CUDA SDK for CUEW; settings are applied per project via orochiApplyCuew().

-- premake keys options by lowercased trigger.
if not premake.option.get("forcecuda") then
    newoption {
        trigger     = "forceCuda",
        description = "Force CUDA backend even if CUDA_PATH is not found (may cause compilation errors)"
    }
end

local function isValidPath(p)
    return p ~= nil and p ~= "" and os.isdir(p)
end

-- Supported CUDA SDK majors, most preferred first; any installed minor is accepted.
local cudaMajors = { 13, 12 }

-- Globbed with forward slashes on every host; premake normalizes them.
local cudaInstallRoots = {
    "/usr/local/cuda-",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v",
}

local function findCudaMajor(major)
    -- The installer's envvar wins so SDKs outside the standard folders are found.
    local fromEnv = os.getenv("CUDA_PATH_V" .. major .. "_0")
    if isValidPath(fromEnv) then
        return fromEnv
    end

    -- Otherwise keep the highest installed minor of this major.
    local bestMinor = -1
    local bestPath = nil
    for _, root in ipairs(cudaInstallRoots) do
        for _, dir in ipairs(os.matchdirs(root .. major .. ".*")) do
            local minor = tonumber(dir:match("%.(%d+)$"))
            if minor ~= nil and minor > bestMinor then
                bestMinor = minor
                bestPath = dir
            end
        end
    end
    return bestPath
end

local function cudaMajorsText()
    return table.concat(cudaMajors, ".x or ") .. ".x"
end

-- Host SDK paths are meaningless for a project generated with --os.
local isCrossGeneration = os.target() ~= os.host()

-- Preferred majors first, then CUDA_PATH, then the default install dir.
local cuda_path = nil
local foundPreferredCudaVersion = false
if not isCrossGeneration then
    for _, major in ipairs(cudaMajors) do
        cuda_path = findCudaMajor(major)
        if isValidPath(cuda_path) then
            break
        end
    end
    foundPreferredCudaVersion = isValidPath(cuda_path)

    if not isValidPath(cuda_path) then
        cuda_path = os.getenv("CUDA_PATH")
    end
    if not isValidPath(cuda_path) and os.isdir("/usr/local/cuda") then
        cuda_path = "/usr/local/cuda"
    end
end

if _ACTION then
    if isValidPath(cuda_path) then
        premake.info("CUEW is enabled. CUDA SDK found: %s", cuda_path)
        if not foundPreferredCudaVersion then
            premake.warn("no supported CUDA version found (%s); using a fallback CUDA SDK install folder.", cudaMajorsText())
        end
    elseif _OPTIONS["forceCuda"] then
        premake.warn("CUEW is force-enabled but CUDA SDK not found (set CUDA_PATH). Compilation may fail.")
    elseif isCrossGeneration then
        premake.warn("CUEW disabled; CUDA SDK detection is skipped when generating for another OS. Use --forceCuda to override.")
    else
        premake.warn("CUEW disabled; CUDA SDK not found (supported: %s). Use --forceCuda to override.", cudaMajorsText())
    end
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
