# =============================================================================
# XSigma
# Compiler-Cache Configuration Module
#
# Provides linalg_target_apply_cache() — a per-target compiler-cache setter. Each Library module
# declares its own XXX_ENABLE_CACHE / XXX_CACHE_BACKEND and calls this function after add_library()
# so that compiler caches are scoped to individual targets rather than globally overriding
# CMAKE_*_COMPILER_LAUNCHER.
#
# Supported backends: none | ccache | sccache | buildcache
#
# Note: compiler caches are safe to apply per-target because they only intercept compilation; they
# never alter flags or output semantics.
# =============================================================================
# include_guard(GLOBAL)

# linalg_target_apply_cache(<target> <enable_var> <backend_var>) enable_var  – boolean CACHE
# variable controlling whether caching is active (e.g. LOGGING_ENABLE_CACHE) backend_var – string
# CACHE variable selecting the cache program (e.g. LOGGING_CACHE_BACKEND:
# none|ccache|sccache|buildcache)
function(linalg_target_apply_cache target_name enable_var backend_var)
    if(NOT ${enable_var})
        message(STATUS "  -cache: ${target_name}: DISABLED")
        return()
    endif()

    set(_backend "${${backend_var}}")

    if(NOT _backend MATCHES "^(none|ccache|sccache|buildcache)$")
        message(FATAL_ERROR "Invalid ${backend_var}: '${_backend}'. "
                            "Valid options are: none, ccache, sccache, buildcache"
        )
    endif()

    if(_backend STREQUAL "none")
        message(STATUS "  -cache: ${target_name}: DISABLED (backend: none)")
        return()
    endif()

    string(TOUPPER "${_backend}" _backend_upper)
    set(_cache_var "LINALG_${_backend_upper}_PROGRAM")
    if(DEFINED XSIGMA_${_backend_upper}_PROGRAM AND NOT DEFINED ${_cache_var})
        set(${_cache_var} "${XSIGMA_${_backend_upper}_PROGRAM}")
    endif()

    if(NOT ${_cache_var})
        find_program(${_cache_var} NAMES ${_backend})
    endif()

    if(${_cache_var} AND EXISTS "${${_cache_var}}")
        message(STATUS "  -cache: ${target_name}: ENABLED. program: ${${_cache_var}}")
        set_property(TARGET "${target_name}" PROPERTY CXX_COMPILER_LAUNCHER "${${_cache_var}}")
        set_property(TARGET "${target_name}" PROPERTY C_COMPILER_LAUNCHER "${${_cache_var}}")
        if(CMAKE_CUDA_COMPILER)
            set_property(TARGET "${target_name}" PROPERTY CUDA_COMPILER_LAUNCHER "${${_cache_var}}")
        endif()
        set_property(TARGET "${target_name}" PROPERTY LINALG_CACHE_BACKEND "${_backend}")
        set_property(TARGET "${target_name}" PROPERTY LINALG_CACHE_EXECUTABLE "${${_cache_var}}")
    else()
        message(
            FATAL_ERROR
                "${target_name}: requested cache ${_backend} not found; install it or select backend none"
        )
    endif()
endfunction()
