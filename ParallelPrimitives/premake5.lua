local rootDir = path.getabsolute("..", _SCRIPT_DIR)

project "ParallelPrimitives"
    kind "StaticLib"

    location "%{wks.location}/%{prj.name}"

    uses { "Orochi" }

    files { "*.h", "*.cpp" }

    -- Bakes into ParallelPrimitives/cache/; the scripts expect the repository root as cwd.
    if _OPTIONS["bakeKernel"] then
        orochiPrebuildScript(rootDir, '"' .. path.join(rootDir, "tools/bakeKernel.bat") .. '"',
                                      'sh "' .. path.join(rootDir, "tools/bakeKernel.sh") .. '"')
    end

    usage "INTERFACE"
        links { "ParallelPrimitives" }
        uses { "Orochi" }
