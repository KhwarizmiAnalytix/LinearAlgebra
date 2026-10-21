# Capture effective target state after all project options and dependencies run.
function(linalg_ci_snapshot)
    set(_snapshot "")
    foreach(_option IN ITEMS LINALG_ENABLE_TESTING LINALG_ENABLE_BLAS LINALG_ENABLE_BLIS
                             LINALG_BLAS_HAS_LAPACKE
    )
        string(APPEND _snapshot "${_option}=${${_option}}\n")
    endforeach()
    foreach(
        _property IN
        ITEMS TYPE
              CXX_STANDARD
              CXX_COMPILER_LAUNCHER
              COMPILE_OPTIONS
              COMPILE_DEFINITIONS
              LINK_OPTIONS
              LINALG_EFFECTIVE_LTO
              LINALG_CACHE_BACKEND
    )
        get_target_property(_value LinearAlgebra "${_property}")
        if(_value MATCHES "-NOTFOUND$")
            set(_value "")
        endif()
        string(APPEND _snapshot "${_property}=${_value}\n")
    endforeach()
    foreach(_target IN ITEMS benchmark_linalg_eigen)
        if(TARGET "${_target}")
            string(APPEND _snapshot "${_target}=ON\n")
        else()
            string(APPEND _snapshot "${_target}=OFF\n")
        endif()
    endforeach()
    file(WRITE "${CMAKE_BINARY_DIR}/ci-effective-flags.txt" "${_snapshot}")
endfunction()
cmake_language(DEFER CALL linalg_ci_snapshot)
