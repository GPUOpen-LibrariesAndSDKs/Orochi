local rootDir = path.getabsolute("..", _SCRIPT_DIR)

project "ParallelPrimitives"
    kind "StaticLib"

    location "%{wks.location}/%{prj.name}"

    useOrochi()

    files { "*.h", "*.cpp" }

    -- The bake scripts write into ParallelPrimitives/cache/, whose contents are
    -- included only by this project, so the step belongs here rather than in the
    -- workspace. Paths inside the scripts are relative to the repository root.
    if _OPTIONS["bakeKernel"] then
        prebuildScript(rootDir, '"' .. path.join(rootDir, "tools/bakeKernel.bat") .. '"',
                                'sh "' .. path.join(rootDir, "tools/bakeKernel.sh") .. '"')
    end
