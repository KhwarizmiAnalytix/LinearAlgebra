# Keep the vendor-neutral provider out of the link interface when MKL is linked.
# A shared provider is required: static archives cannot be opened with dlopen.
set(_LINALG_ISOLATED_BLAS_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(linalg_configure_isolated_blas)
    if(LINALG_ENABLE_SANITIZER AND UNIX AND NOT APPLE)
        message(
            FATAL_ERROR
                "Combined MKL/BLAS builds use RTLD_DEEPBIND, which is incompatible with sanitizers. "
                "Run sanitizer checks with one CPU vendor backend enabled at a time."
        )
    endif()
    if(UNIX AND NOT APPLE)
        check_cxx_symbol_exists(RTLD_DEEPBIND "dlfcn.h" LINALG_HAVE_RTLD_DEEPBIND)
        if(NOT LINALG_HAVE_RTLD_DEEPBIND)
            message(FATAL_ERROR "Combined MKL/BLAS builds require RTLD_DEEPBIND on this platform.")
        endif()
    endif()

    # Use the selected headers' prefixes as well as normal CMake search paths.
    set(_runtime_hints "")
    foreach(_include_dir IN ITEMS "${LINALG_CBLAS_INCLUDE_DIR}" "${LINALG_LAPACKE_INCLUDE_DIR}")
        if(_include_dir)
            get_filename_component(_prefix "${_include_dir}/.." ABSOLUTE)
            list(APPEND _runtime_hints "${_prefix}" "${_prefix}/..")
        endif()
    endforeach()
    if(WIN32)
        find_file(
            LINALG_BLAS_RUNTIME_LIBRARY
            NAMES libopenblas.dll openblas.dll libblis.dll blis.dll
            HINTS ${_runtime_hints}
            PATH_SUFFIXES bin lib
        )
    else()
        # Temporarily exclude static archives from find_library's candidates.
        if(APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib .so)
        else()
            set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
        endif()
        find_library(
            LINALG_BLAS_RUNTIME_LIBRARY
            NAMES openblas blis
            HINTS ${_runtime_hints}
            PATH_SUFFIXES lib lib64
        )
    endif()
    set(LINALG_BLAS_RUNTIME_LIBRARY "${LINALG_BLAS_RUNTIME_LIBRARY}"
        CACHE FILEPATH "Shared CBLAS provider loaded independently of MKL"
    )
    # linalg_check_blas_provider() only sets LINALG_LAPACKE_LIBRARY when LAPACKE lives in a
    # library separate from BLAS_LIBRARIES (e.g. Debian/Ubuntu's liblapacke.so, distinct from
    # libopenblas.so). Re-resolve it the same shared-only way as LINALG_BLAS_RUNTIME_LIBRARY above,
    # so resolve_blas_symbol() looks LAPACKE_* up in the library that actually exports it instead
    # of always falling back to LINALG_BLAS_RUNTIME_LIBRARY.
    set(_linalg_lapacke_runtime_default "")
    if(LINALG_LAPACKE_LIBRARY AND NOT LINALG_LAPACKE_LIBRARY STREQUAL LINALG_BLAS_RUNTIME_LIBRARY)
        if(WIN32)
            find_file(
                _linalg_lapacke_runtime_default
                NAMES liblapacke.dll lapacke.dll
                HINTS ${_runtime_hints}
                PATH_SUFFIXES bin lib
            )
        else()
            get_filename_component(_linalg_lapacke_library_dir "${LINALG_LAPACKE_LIBRARY}" DIRECTORY)
            find_library(
                _linalg_lapacke_runtime_default
                NAMES lapacke
                HINTS ${_runtime_hints} "${_linalg_lapacke_library_dir}"
                PATH_SUFFIXES lib lib64
            )
            unset(_linalg_lapacke_library_dir)
        endif()
        if(NOT _linalg_lapacke_runtime_default)
            # Fall back to the configure-time library even if it did not match the
            # shared-only search above (e.g. a non-standard name find_library missed).
            set(_linalg_lapacke_runtime_default "${LINALG_LAPACKE_LIBRARY}")
        endif()
    endif()
    set(LINALG_LAPACKE_RUNTIME_LIBRARY "${_linalg_lapacke_runtime_default}"
        CACHE FILEPATH
              "Separate shared LAPACKE provider, or empty when bundled with the CBLAS provider"
    )
    unset(_linalg_lapacke_runtime_default)
    if(NOT EXISTS "${LINALG_BLAS_RUNTIME_LIBRARY}")
        message(
            FATAL_ERROR
                "MKL + BLAS requires an independent shared BLAS provider. "
                "Set LINALG_BLAS_RUNTIME_LIBRARY to its .dll/.so/.dylib and "
                "LINALG_CBLAS_INCLUDE_DIR / LINALG_LAPACKE_INCLUDE_DIR to its matching headers."
        )
    endif()
    if(LINALG_LAPACKE_RUNTIME_LIBRARY AND NOT EXISTS "${LINALG_LAPACKE_RUNTIME_LIBRARY}")
        message(FATAL_ERROR "LINALG_LAPACKE_RUNTIME_LIBRARY does not exist.")
    endif()
    if(LINALG_BLAS_USE_ACCELERATE)
        message(
            FATAL_ERROR
                "Combined MKL/BLAS builds require standalone CBLAS headers and a shared provider."
        )
    endif()
    configure_file(
        "${_LINALG_ISOLATED_BLAS_MODULE_DIR}/isolated_blas_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/isolated_blas_config.h" @ONLY
    )
    message(STATUS "LinearAlgebra: independent BLAS runtime: ${LINALG_BLAS_RUNTIME_LIBRARY}")
    if(LINALG_LAPACKE_RUNTIME_LIBRARY)
        message(
            STATUS "LinearAlgebra: independent LAPACKE runtime: ${LINALG_LAPACKE_RUNTIME_LIBRARY}"
        )
    endif()
endfunction()
