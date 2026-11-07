# =============================================================================
# rocm_prefix.cmake — ROCm / HIP SDK prefix detection for LinearAlgebra.
#
# hip-config.cmake lives under /opt/rocm (or $ROCM_PATH), which is not a
# CMake default search path. Debian/Ubuntu also install /usr/bin/hipcc as an
# alternatives wrapper; taking dirname(hipcc) yields HIP_PATH=/usr, which is
# not an SDK root and causes find_package(hip) to fail.
#
# Ported from Vectorization's Cmake/tools/rocm_prefix.cmake via Scalars'
# cmake/rocm_prefix.cmake (scalars_setup_rocm_prefix → linalg_setup_rocm_prefix).
# =============================================================================
include_guard(GLOBAL)

# linalg_setup_rocm_prefix()
#
# Prepends the first valid ROCm SDK root to CMAKE_PREFIX_PATH and corrects
# HIP_PATH / ROCM_PATH env vars when they point at /usr. Idempotent: sets
# LINALG_ROCM_PREFIX on success (no-op if already set).
macro(linalg_setup_rocm_prefix)
    if(NOT LINALG_ROCM_PREFIX)
        set(_linalg_rocm_roots "")
        foreach(_cand IN ITEMS "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" "/opt/rocm")
            if(_cand
               AND NOT _cand STREQUAL "/usr"
               AND NOT _cand STREQUAL "/usr/local"
               AND NOT _cand STREQUAL "/"
            )
                list(APPEND _linalg_rocm_roots "${_cand}")
            endif()
        endforeach()
        file(GLOB _linalg_rocm_versioned "/opt/rocm-*")
        list(APPEND _linalg_rocm_roots ${_linalg_rocm_versioned})

        set(_linalg_rocm_found "")
        foreach(_root IN LISTS _linalg_rocm_roots)
            if(IS_DIRECTORY "${_root}")
                if(EXISTS "${_root}/lib/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/lib64/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/hip/lib/cmake/hip/hip-config.cmake"
                   OR EXISTS "${_root}/include/hip/hip_runtime.h"
                   OR EXISTS "${_root}/hip/include/hip/hip_runtime.h"
                )
                    set(_linalg_rocm_found "${_root}")
                    break()
                endif()
            endif()
        endforeach()

        if(_linalg_rocm_found)
            list(PREPEND CMAKE_PREFIX_PATH "${_linalg_rocm_found}")
            list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
            set(LINALG_ROCM_PREFIX "${_linalg_rocm_found}")
            if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "/usr")
                set(ENV{ROCM_PATH} "${_linalg_rocm_found}")
            endif()
            if(NOT DEFINED ENV{HIP_PATH} OR "$ENV{HIP_PATH}" STREQUAL "/usr")
                set(ENV{HIP_PATH} "${_linalg_rocm_found}")
            endif()
            message(STATUS "LinearAlgebra: ROCm/HIP SDK root: ${_linalg_rocm_found}")
            # hip-config.cmake reads AMDGPU_TARGETS / GPU_TARGETS. Without them it
            # runs amdgpu-arch, which fails on headless build machines and falls back
            # to gfx906. Set them from LINALG_HIP_ARCHITECTURES when available so
            # hip-config.cmake targets the right GPU families without a physical GPU.
            if(DEFINED LINALG_HIP_ARCHITECTURES AND NOT LINALG_HIP_ARCHITECTURES STREQUAL "native")
                if(NOT AMDGPU_TARGETS)
                    set(AMDGPU_TARGETS "${LINALG_HIP_ARCHITECTURES}")
                endif()
                if(NOT GPU_TARGETS)
                    set(GPU_TARGETS "${LINALG_HIP_ARCHITECTURES}")
                endif()
            endif()
        endif()
        unset(_linalg_rocm_roots)
        unset(_linalg_rocm_versioned)
        unset(_linalg_rocm_found)
        unset(_cand)
        unset(_root)
    endif()
endmacro()
