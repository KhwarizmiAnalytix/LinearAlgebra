# Validate declarations and capabilities against the selected provider, never against MKL.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

macro(linalg_check_blas_provider)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${_linalg_blas_libraries})
    set(_linalg_saved_find_framework "${CMAKE_FIND_FRAMEWORK}")
    set(CMAKE_FIND_FRAMEWORK NEVER)
    # Re-run capability probes when users switch vendors or headers in the same build tree.
    unset(LINALG_CBLAS_LINKS CACHE)
    unset(LINALG_LAPACKE_LINKS CACHE)
    if(BLA_VENDOR STREQUAL "Apple" OR "${BLAS_LIBRARIES}" MATCHES "Accelerate")
        set(LINALG_BLAS_USE_ACCELERATE ON)
        set(LINALG_BLAS_VENDOR Apple)
        set(_linalg_cblas_header "Accelerate/Accelerate.h")
    else()
        set(LINALG_BLAS_VENDOR "${BLA_VENDOR}")
        if("${BLAS_LIBRARIES}" MATCHES "[Oo][Pp][Ee][Nn][Bb][Ll][Aa][Ss]")
            set(LINALG_BLAS_VENDOR OpenBLAS)
        endif()
        # Prefer headers alongside the libraries actually found by FindBLAS.
        set(_linalg_provider_prefixes "")
        foreach(_lib IN LISTS BLAS_LIBRARIES BLIS_LIBRARIES)
            if(IS_ABSOLUTE "${_lib}")
                get_filename_component(_dir "${_lib}" DIRECTORY)
                list(APPEND _linalg_provider_prefixes "${_dir}/.." "${_dir}/../..")
            endif()
        endforeach()
        if(TARGET BLIS::BLIS)
            get_target_property(_blis_includes BLIS::BLIS INTERFACE_INCLUDE_DIRECTORIES)
            if(_blis_includes)
                list(APPEND LINALG_BLAS_INCLUDE_DIRS ${_blis_includes})
            endif()
        endif()
        if(NOT LINALG_CBLAS_INCLUDE_DIR)
            unset(_linalg_cblas_include)
            find_path(
                _linalg_cblas_include
                NAMES cblas.h NO_CACHE
                HINTS ${LINALG_BLAS_INCLUDE_DIRS} ${_linalg_provider_prefixes}
                PATH_SUFFIXES
                    include
                    include/openblas
                    include/blis
                    "include/${CMAKE_LIBRARY_ARCHITECTURE}"
                    "include/${CMAKE_LIBRARY_ARCHITECTURE}/openblas"
                    openblas
                    blis
                NO_DEFAULT_PATH
            )
            if(NOT _linalg_cblas_include AND NOT _linalg_provider_prefixes)
                # Some finders return linker names rather than absolute library paths.
                find_path(_linalg_cblas_include NAMES cblas.h NO_CACHE PATH_SUFFIXES openblas blis)
            endif()
            set(LINALG_CBLAS_INCLUDE_DIR "${_linalg_cblas_include}")
        endif()
        if(LINALG_CBLAS_INCLUDE_DIR MATCHES "[.]framework")
            message(
                FATAL_ERROR
                    "A non-Apple BLAS provider requires its own cblas.h; clear LINALG_CBLAS_INCLUDE_DIR"
            )
        endif()
        if(NOT LINALG_CBLAS_INCLUDE_DIR)
            message(
                FATAL_ERROR
                    "Selected BLAS provider requires matching cblas.h; set LINALG_CBLAS_INCLUDE_DIR"
            )
        endif()
        list(APPEND LINALG_BLAS_INCLUDE_DIRS "${LINALG_CBLAS_INCLUDE_DIR}")
        set(CMAKE_REQUIRED_INCLUDES ${LINALG_BLAS_INCLUDE_DIRS})
        set(_linalg_cblas_header "cblas.h")
    endif()
    check_cxx_source_compiles(
        "
        #include <${_linalg_cblas_header}>
        int main() {
            void (*gemm)(decltype(CblasRowMajor), decltype(CblasNoTrans), decltype(CblasNoTrans),
                int, int, int, float, const float*, int, const float*, int, float, float*, int)
                = &cblas_sgemm;
            float a = 1; gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1, &a, 1, &a, 1, 0, &a, 1);
        }"
        LINALG_CBLAS_LINKS
    )
    if(NOT LINALG_CBLAS_LINKS)
        message(
            FATAL_ERROR
                "Selected BLAS libraries and CBLAS headers do not provide a matching LP64 interface"
        )
    endif()
    # Accelerate intentionally has no LAPACKE; unrelated installed headers must not enable it.
    if(NOT LINALG_BLAS_USE_ACCELERATE)
        if(NOT LINALG_LAPACKE_INCLUDE_DIR)
            unset(_linalg_lapacke_include)
            find_path(
                _linalg_lapacke_include
                NAMES lapacke.h NO_CACHE
                HINTS "${LINALG_CBLAS_INCLUDE_DIR}" ${_linalg_provider_prefixes}
                PATH_SUFFIXES include include/openblas
            )
            set(LINALG_LAPACKE_INCLUDE_DIR "${_linalg_lapacke_include}")
        endif()
        if(LINALG_LAPACKE_INCLUDE_DIR)
            list(APPEND CMAKE_REQUIRED_INCLUDES "${LINALG_LAPACKE_INCLUDE_DIR}")
            if(MSVC)
                set(CMAKE_REQUIRED_DEFINITIONS "-Dlapack_complex_float=std::complex<float>"
                                               "-Dlapack_complex_double=std::complex<double>"
                )
            endif()
            set(_linalg_lapacke_probe
                "
                #include <complex>
                #include <lapacke.h>
                static_assert(sizeof(lapack_int) == 4, \"LAPACKE must be LP64\");
                int main() { float a = 1; lapack_int pivot;
                    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, 1, 1, &a, 1, &pivot); }"
            )
            check_cxx_source_compiles("${_linalg_lapacke_probe}" LINALG_LAPACKE_LINKS)
            if(NOT LINALG_LAPACKE_LINKS)
                find_library(
                    LINALG_LAPACKE_LIBRARY
                    NAMES lapacke
                    HINTS ${_linalg_provider_prefixes}
                    PATH_SUFFIXES lib lib64
                )
                if(LINALG_LAPACKE_LIBRARY)
                    list(APPEND CMAKE_REQUIRED_LIBRARIES "${LINALG_LAPACKE_LIBRARY}")
                    unset(LINALG_LAPACKE_LINKS CACHE)
                    check_cxx_source_compiles("${_linalg_lapacke_probe}" LINALG_LAPACKE_LINKS)
                    if(LINALG_LAPACKE_LINKS)
                        list(APPEND _linalg_blas_libraries "${LINALG_LAPACKE_LIBRARY}")
                    endif()
                endif()
            endif()
            if(LINALG_LAPACKE_LINKS)
                list(APPEND LINALG_BLAS_INCLUDE_DIRS "${LINALG_LAPACKE_INCLUDE_DIR}")
                set(LINALG_BLAS_HAS_LAPACKE ON)
            endif()
        endif()
    endif()
    if(NOT LINALG_BLAS_HAS_LAPACKE)
        message(
            STATUS
                "Selected BLAS provider has no usable LAPACKE; factorization operations use scalar fallbacks"
        )
    endif()
    list(REMOVE_DUPLICATES LINALG_BLAS_INCLUDE_DIRS)
    set(CMAKE_FIND_FRAMEWORK "${_linalg_saved_find_framework}")
    cmake_pop_check_state()
endmacro()
