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

-- Supported CUDA SDK majors, in order of preference. Any minor of these majors is accepted:
-- the install directories are globbed, so a new 13.x or 12.x release is picked up without editing this list.
local cudaMajors = { "13", "12" }

local function findCudaVersion(major)
    -- An envvar set by the installer wins, so a SDK outside the standard folders is still found.
    local envPath = os.getenv("CUDA_PATH_V" .. major .. "_0")
    if isValidPath(envPath) then
        return envPath
    end

    -- Otherwise look for any installed minor of this major and keep the highest one.
    local bestMinor = -1
    local bestPath = nil
    for _, root in ipairs({ "/usr/local/cuda-", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v" }) do
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

-- Preferred majors first, then CUDA_PATH, then the default install dir.
local cuda_path = nil
for _, major in ipairs(cudaMajors) do
    cuda_path = findCudaVersion(major)
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
local cudaVersionsText = table.concat(cudaMajors, ".x, ") .. ".x"
if isValidPath(cuda_path) then
    print("CUEW is enabled. CUDA SDK found: " .. cuda_path)
    if not foundPreferredCudaVersion then
        print("WARNING: preferred CUDA version not found (" .. cudaVersionsText .. "); using a fallback CUDA SDK install folder.")
    end
elseif _OPTIONS["forceCuda"] then
    print("WARNING: CUEW is force-enabled but CUDA SDK not found (set CUDA_PATH). Compilation may fail.")
else
    print("WARNING: CUEW disabled; CUDA SDK not found (preferred: " .. cudaVersionsText .. "). Use --forceCuda to override.")
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
