local rootDir = path.getabsolute("..", _SCRIPT_DIR)

project "ParallelPrimitives"
    kind "StaticLib"

    location "%{wks.location}/%{prj.name}"

    uses { "Orochi" }

    files { "*.h", "*.cpp" }

    -- Bakes into ParallelPrimitives/cache/; the scripts expect the repository root as cwd.
    if _OPTIONS["bakeKernel"] then
        local cacheDir = path.join(rootDir, "ParallelPrimitives/cache")
        filter { "files:RadixSortKernels.h", "system:windows" }
            buildcommands { 'pushd "' .. rootDir .. '" && "' .. path.join(rootDir, "tools/bakeKernel.bat") .. '" && popd' }
        filter { "files:RadixSortKernels.h", "system:not windows" }
            buildcommands { '( cd "' .. rootDir .. '" && sh "' .. path.join(rootDir, "tools/bakeKernel.sh") .. '" )' }
        filter "files:RadixSortKernels.h"
            buildmessage "Baking ParallelPrimitives kernels"
            buildinputs { path.join(rootDir, "ParallelPrimitives/RadixSortConfigs.h") }
            buildoutputs { path.join(cacheDir, "Kernels.h"), path.join(cacheDir, "KernelArgs.h") }
        filter {}
    end

    usage "PUBLIC"
        uses { "Orochi" }

    usage "INTERFACE"
        links { "ParallelPrimitives" }
