# =============================================================================
# XSigma Helper
# Macros Module

# This module provides utility macros for common CMake operations used throughout the XSigma build
# system, including source file filtering and organization.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Use Icecream (icecc) as the compiler launcher for a single target when XXX_ENABLE_ICECC is ON.
function(linalg_target_optional_icecc target_name enable_var)
    if(NOT ${enable_var})
        return()
    endif()
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "linalg_target_optional_icecc: target '${target_name}' does not exist")
    endif()
    if(NOT LINALG_ICECC_EXECUTABLE)
        find_program(LINALG_ICECC_EXECUTABLE NAMES icecc)
    endif()
    if(LINALG_ICECC_EXECUTABLE)
        get_target_property(_launcher "${target_name}" CXX_COMPILER_LAUNCHER)
        get_target_property(_cache_backend "${target_name}" LINALG_CACHE_BACKEND)
        get_target_property(_cache_program "${target_name}" LINALG_CACHE_EXECUTABLE)
        string(REPLACE "'" "'\"'\"'" _icecc_shell "${LINALG_ICECC_EXECUTABLE}")
        if(_launcher AND _cache_backend MATCHES "^(ccache|sccache)$")
            # Caches must see a compiler, rather than `icecc` as the compiler.
            # The compiler wrapper delegates its compiler-compatible interface to
            # Icecream. Fingerprint both underlying tools so cache keys change
            # when either executable changes, even though caches hash a script.
            file(SHA256 "${LINALG_ICECC_EXECUTABLE}" _icecc_hash)
            string(REPLACE "'" "'\"'\"'" _cache_shell "${_cache_program}")
            set(_wrapper_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target_name}-launchers")
            file(MAKE_DIRECTORY "${_wrapper_dir}")
            foreach(_language IN ITEMS C CXX)
                if(NOT CMAKE_${_language}_COMPILER_LOADED)
                    continue()
                endif()
                set(_compiler "${CMAKE_${_language}_COMPILER}")
                file(SHA256 "${_compiler}" _compiler_hash)
                set_property(
                    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_compiler}"
                                              "${LINALG_ICECC_EXECUTABLE}"
                )
                string(REPLACE "'" "'\"'\"'" _compiler_shell "${_compiler}")
                set(_wrapper "${_wrapper_dir}/icecc-${_language}-compiler")
                string(REPLACE "'" "'\"'\"'" _wrapper_shell "${_wrapper}")
                set(_cache_launcher "${_wrapper_dir}/icecc-${_cache_backend}-${_language}-launcher")
                configure_file(
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/icecc_compiler.sh.in"
                    "${_wrapper}"
                    @ONLY
                    FILE_PERMISSIONS
                    OWNER_READ
                    OWNER_WRITE
                    OWNER_EXECUTE
                    GROUP_READ
                    GROUP_EXECUTE
                    WORLD_READ
                    WORLD_EXECUTE
                )
                configure_file(
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/icecc_cache_launcher.sh.in"
                    "${_cache_launcher}"
                    @ONLY
                    FILE_PERMISSIONS
                    OWNER_READ
                    OWNER_WRITE
                    OWNER_EXECUTE
                    GROUP_READ
                    GROUP_EXECUTE
                    WORLD_READ
                    WORLD_EXECUTE
                )
                set_property(
                    TARGET "${target_name}" PROPERTY ${_language}_COMPILER_LAUNCHER
                                                     "${_cache_launcher}"
                )
            endforeach()
            message(STATUS "${target_name}: ${_cache_backend} with Icecream on cache misses")
            return()
        elseif(_launcher)
            message(FATAL_ERROR "Icecream cannot compose with this compiler launcher: ${_launcher}")
        endif()
        set_property(
            TARGET "${target_name}" PROPERTY CXX_COMPILER_LAUNCHER "${LINALG_ICECC_EXECUTABLE}"
        )
        set_property(
            TARGET "${target_name}" PROPERTY C_COMPILER_LAUNCHER "${LINALG_ICECC_EXECUTABLE}"
        )
        message(STATUS "Icecream compiler launcher for ${target_name}: ${LINALG_ICECC_EXECUTABLE}")
    else()
        message(STATUS "Icecream enabled for ${target_name} but icecc was not found in PATH")
    endif()
endfunction()

macro(linalg_module_remove_underscores name_in name_out)
    string(REPLACE "_" ";" name_splited ${name_in})
    set(the_list "")
    foreach(name IN ITEMS ${name_splited})
        string(SUBSTRING ${name} 0 1 first_letter)
        string(TOUPPER ${first_letter} first_letter)
        string(REGEX REPLACE "^.(.*)" "${first_letter}\\1" result "${name}")
        list(APPEND the_list ${result})
    endforeach()
    string(REPLACE ";" "" name_splited_merge ${the_list})
    string(SUBSTRING ${name_splited_merge} 0 1 first_letter)
    string(TOLOWER ${first_letter} first_letter)
    string(REGEX REPLACE "^.(.*)" "${first_letter}\\1" ${name_out} "${name_splited_merge}")
endmacro()
