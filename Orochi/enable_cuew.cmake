# Enables CUEW (CUDA Extension Wrangler) when the CUDA SDK is available.
#
# Mirrors the search policy of enable_cuew.lua -- keep both in sync.
#
# In order to have Orochi compiled with CUDA, you need to define
# OROCHI_ENABLE_CUEW and add the CUDA include path to your Orochi project.
# If your project is using cmake, this script can be included:
#   include(${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)

option(FORCE_CUDA "Force the CUDA backend even if the CUDA SDK is not found" OFF)

# Most preferred first.
set(OROCHI_CUDA_VERSIONS "12.2")

function(_orochi_find_cuda_version result version)
    string(REPLACE "." "_" versionSuffix "${version}")
    set(candidates
        "$ENV{CUDA_PATH_V${versionSuffix}}"
        "/usr/local/cuda-${version}"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}"
    )
    foreach(candidate IN LISTS candidates)
        if(candidate AND IS_DIRECTORY "${candidate}")
            set(${result} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${result} "" PARENT_SCOPE)
endfunction()

# Preferred versions first, then CUDA_PATH, then the default install dir.
set(cuda_path "")
foreach(version IN LISTS OROCHI_CUDA_VERSIONS)
    _orochi_find_cuda_version(cuda_path "${version}")
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

string(REPLACE ";" ", " cudaVersionsText "${OROCHI_CUDA_VERSIONS}")

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
