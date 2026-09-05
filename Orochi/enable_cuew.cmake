# Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.
#
# Mirrors the search policy of enable_cuew.lua -- keep both in sync.
#
# In order to have Orochi compiled with CUDA, you need to define
# OROCHI_ENABLE_CUEW and add the CUDA include path to your Orochi project.
# If your project is using cmake, this script can be included:
#   include(${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)

option(FORCE_CUDA "Force the CUDA backend even if the CUDA SDK is not found" OFF)

# Supported CUDA SDK majors, most preferred first. Any minor of these majors is
# accepted: the install directories are globbed, so a new 13.x or 12.x release
# is picked up without editing this list.
set(OROCHI_CUDA_MAJORS 13 12)

set(OROCHI_CUDA_INSTALL_ROOTS
    "/usr/local/cuda-"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v"
)

function(_orochi_find_cuda_major result major)
    # An envvar set by the installer wins, so an SDK outside the standard
    # install folders is still found.
    set(fromEnv "$ENV{CUDA_PATH_V${major}_0}")
    if(fromEnv AND IS_DIRECTORY "${fromEnv}")
        set(${result} "${fromEnv}" PARENT_SCOPE)
        return()
    endif()

    # Otherwise keep the highest installed minor of this major.
    set(bestMinor -1)
    set(bestPath "")
    foreach(root IN LISTS OROCHI_CUDA_INSTALL_ROOTS)
        file(GLOB candidates "${root}${major}.*")
        foreach(candidate IN LISTS candidates)
            if(IS_DIRECTORY "${candidate}" AND candidate MATCHES "\\.([0-9]+)$" AND CMAKE_MATCH_1 GREATER bestMinor)
                set(bestMinor ${CMAKE_MATCH_1})
                set(bestPath "${candidate}")
            endif()
        endforeach()
    endforeach()
    set(${result} "${bestPath}" PARENT_SCOPE)
endfunction()

# Preferred majors first, then CUDA_PATH, then the default install dir.
# This file is include()d into the caller's scope, so start from a known state
# rather than inheriting whatever the including project left in cuda_path.
set(cuda_path "")
foreach(major IN LISTS OROCHI_CUDA_MAJORS)
    _orochi_find_cuda_major(cuda_path "${major}")
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

string(REPLACE ";" ".x or " cudaMajorsText "${OROCHI_CUDA_MAJORS}")
set(cudaMajorsText "${cudaMajorsText}.x")

if(cuda_path)
    message(STATUS "CUEW is enabled. CUDA SDK found: ${cuda_path}")
    add_compile_definitions(OROCHI_ENABLE_CUEW)
    if(NOT foundPreferredCudaVersion)
        message(WARNING "no supported CUDA version found (${cudaMajorsText}); using a fallback CUDA SDK install folder.")
    endif()
    include_directories(SYSTEM "${cuda_path}/include")
elseif(FORCE_CUDA)
    message(WARNING "CUEW is force-enabled but CUDA SDK not found (set CUDA_PATH). Compilation may fail.")
    add_compile_definitions(OROCHI_ENABLE_CUEW)
else()
    message(WARNING "CUEW disabled; CUDA SDK not found (supported: ${cudaMajorsText}). Set FORCE_CUDA=ON to override.")
endif()
