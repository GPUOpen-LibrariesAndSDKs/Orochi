# Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.
#
# Mirrors the search policy of enable_cuew.lua -- keep both in sync.
#
# In order to have Orochi compiled with CUDA, you need to define
# OROCHI_ENABLE_CUEW and add the CUDA include path to your Orochi project.
# If your project is using cmake, this script can be included:
#   include(${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)

option(FORCE_CUDA "Force the CUDA backend even if the CUDA SDK is not found" OFF)

# Supported CUDA SDK majors, in order of preference. Any minor of these majors is accepted:
# the install directories are globbed, so a new 13.x or 12.x release is picked up without editing this list.
set(OROCHI_CUDA_MAJORS "13" "12")

function(_orochi_find_cuda_version result major)
    # An envvar set by the installer wins, so a SDK outside the standard folders is still found.
    if(DEFINED ENV{CUDA_PATH_V${major}_0} AND IS_DIRECTORY "$ENV{CUDA_PATH_V${major}_0}")
        set(${result} "$ENV{CUDA_PATH_V${major}_0}" PARENT_SCOPE)
        return()
    endif()

    # Otherwise look for any installed minor of this major and keep the highest one.
    set(best_minor -1)
    set(best_path "")
    foreach(root "/usr/local/cuda-" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v")
        file(GLOB candidates "${root}${major}.*")
        foreach(candidate IN LISTS candidates)
            if(IS_DIRECTORY "${candidate}" AND candidate MATCHES "\\.([0-9]+)$" AND CMAKE_MATCH_1 GREATER best_minor)
                set(best_minor "${CMAKE_MATCH_1}")
                set(best_path "${candidate}")
            endif()
        endforeach()
    endforeach()
    set(${result} "${best_path}" PARENT_SCOPE)
endfunction()

# Preferred majors first, then CUDA_PATH, then the default install dir.
set(cuda_path "")
foreach(major IN LISTS OROCHI_CUDA_MAJORS)
    _orochi_find_cuda_version(cuda_path "${major}")
    if(cuda_path)
        break()
    endif()
endforeach()
set(foundPreferredCudaVersion "${cuda_path}")

if(NOT cuda_path AND IS_DIRECTORY "$ENV{CUDA_PATH}")
    set(cuda_path "$ENV{CUDA_PATH}")
endif()
if(NOT cuda_path AND IS_DIRECTORY "/usr/local/cuda")
    set(cuda_path "/usr/local/cuda")
endif()

string(REPLACE ";" ".x, " cudaVersionsText "${OROCHI_CUDA_MAJORS}")
string(APPEND cudaVersionsText ".x")

if(cuda_path)
    message(STATUS "CUEW is enabled. CUDA SDK found: ${cuda_path}")
    add_compile_definitions(OROCHI_ENABLE_CUEW)
    if(NOT foundPreferredCudaVersion)
        message(WARNING "preferred CUDA version not found (${cudaVersionsText}); using a fallback CUDA SDK install folder.")
    endif()
    include_directories(SYSTEM "${cuda_path}/include")
elseif(FORCE_CUDA)
    message(WARNING "CUEW is force-enabled but CUDA SDK not found (set CUDA_PATH). Compilation may fail.")
    add_compile_definitions(OROCHI_ENABLE_CUEW)
else()
    message(WARNING "CUEW disabled; CUDA SDK not found (preferred: ${cudaVersionsText}). Set FORCE_CUDA=ON to override.")
endif()
