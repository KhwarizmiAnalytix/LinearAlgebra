if(LINALG_STANDALONE)
    message("")
    message("******** LinearAlgebra module (CMake flags) ********")
    message("  Shared libs          : ${BUILD_SHARED_LIBS}")
    message("  Cxx standard         : ${LINALG_CXX_STANDARD}")
    message("  Cache                : ${LINALG_ENABLE_CACHE}")
    message("  Cache backend        : ${LINALG_CACHE_BACKEND}")
    message("  Icecc                : ${LINALG_ENABLE_ICECC}")
    linalg_print_lto_linker_summary(LINALG)
    message("  Coverage             : ${LINALG_ENABLE_COVERAGE}")
    message("  Testing              : ${LINALG_ENABLE_TESTING}")
    message("  CTest dashboards     : ${BUILD_TESTING}")
    message("  Gtest                : ${LINALG_ENABLE_GTEST}")
    message("  Examples             : ${LINALG_BUILD_EXAMPLES}")
    message("  Benchmark            : ${LINALG_ENABLE_BENCHMARK}")
    message("  Profiler             : ${LINALG_ENABLE_PROFILING}")
    message("  MKL                  : ${LINALG_ENABLE_MKL}")
    message("  BLAS/LAPACK          : ${LINALG_ENABLE_BLAS}")
    message("  BLIS                 : ${LINALG_ENABLE_BLIS}")
    message("  Accelerate (BLAS)    : ${LINALG_BLAS_USE_ACCELERATE}")
    message("  LAPACKE (BLAS)       : ${LINALG_BLAS_HAS_LAPACKE}")
    message("  CUDA (cuBLAS)       : ${LINALG_ENABLE_CUBLAS}")
    message("  HIP (hipBLAS)       : ${LINALG_ENABLE_ROCBLAS}")
    message("  Metal (MPS)         : ${LINALG_ENABLE_METAL}")
    message("  Cpuinfo              : ${LINALG_HAS_CPUINFO}")
    message("  LU pivoting          : ${LINALG_LU_PIVOTING}")
    message("  Win32 warnings       : ${LINALG_DISPLAY_WIN32_WARNINGS}")
    message("  Vectorization        : ${LINALG_ENABLE_VECTORIZATION}")
    message("  Vectorization ISA    : ${LINALG_VECTORIZATION_ISA}")
    message("  Clang-tidy           : ${LINALG_ENABLE_CLANGTIDY}")
    message("  Fix                  : ${LINALG_ENABLE_FIX}")
    message("  Iwyu                 : ${LINALG_ENABLE_IWYU}")
    message("  Sanitizer            : ${LINALG_ENABLE_SANITIZER}")
    message("  Sanitizer type       : ${LINALG_SANITIZER_TYPE}")
    message("  Spell                : ${LINALG_ENABLE_SPELL}")
    message("  Valgrind             : ${LINALG_ENABLE_VALGRIND}")
    message("  External dependencies: ${LINALG_ENABLE_EXTERNAL}")
    message("  Third-party sources  : ${LINALG_THIRD_PARTY_DIR}")
    message("")
    message("Build:")
    message("  Build type           : ${CMAKE_BUILD_TYPE}")
    message("  Generator            : ${CMAKE_GENERATOR}")
    message("  C compiler           : ${CMAKE_C_COMPILER}")
    message("  CXX compiler         : ${CMAKE_CXX_COMPILER}")
    message("  Default C++ standard : ${CMAKE_CXX_STANDARD}")
    message("  Export compile cmds  : ${CMAKE_EXPORT_COMPILE_COMMANDS}")
    message("  Install prefix       : ${CMAKE_INSTALL_PREFIX}")
    message("  C flags              : ${CMAKE_C_FLAGS}")
    message("  CXX flags            : ${CMAKE_CXX_FLAGS}")
    message("  Exe linker flags     : ${CMAKE_EXE_LINKER_FLAGS}")
    message("  Shared linker flags  : ${CMAKE_SHARED_LINKER_FLAGS}")
    message("  Static linker flags  : ${CMAKE_STATIC_LINKER_FLAGS}")
    message("  Module linker flags  : ${CMAKE_MODULE_LINKER_FLAGS}")
    # Include configuration-specific flags for both single- and multi-config generators.
    set(_summary_configs ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
    list(REMOVE_DUPLICATES _summary_configs)
    foreach(_summary_config IN LISTS _summary_configs)
        string(TOUPPER "${_summary_config}" _summary_config_upper)
        foreach(
            _summary_flag IN
            ITEMS CMAKE_C_FLAGS
                  CMAKE_CXX_FLAGS
                  CMAKE_CUDA_FLAGS
                  CMAKE_EXE_LINKER_FLAGS
                  CMAKE_SHARED_LINKER_FLAGS
                  CMAKE_STATIC_LINKER_FLAGS
                  CMAKE_MODULE_LINKER_FLAGS
        )
            set(_summary_var "${_summary_flag}_${_summary_config_upper}")
            if(DEFINED ${_summary_var})
                message("  ${_summary_var} : ${${_summary_var}}")
            endif()
        endforeach()
    endforeach()
    get_directory_property(_summary_compile_defs DIRECTORY ${LINALG_SOURCE_DIR} COMPILE_DEFINITIONS)
    message("  Top-level compile defs : ${_summary_compile_defs}")
    message("")

    # Optional overrides and tool paths from the Cmake/ helpers. Keep unset
    # platform/backend-specific settings out of the summary.
    message("Additional CMake settings (when defined):")
    foreach(
        _summary_var IN
        ITEMS CMAKE_CONFIGURATION_TYPES
              CMAKE_GENERATOR_PLATFORM
              CMAKE_GENERATOR_TOOLSET
              CMAKE_TOOLCHAIN_FILE
              CMAKE_MAKE_PROGRAM
              CMAKE_LINKER
              CMAKE_PREFIX_PATH
              CMAKE_FIND_ROOT_PATH
              CMAKE_INTERPROCEDURAL_OPTIMIZATION
              CMAKE_MSVC_RUNTIME_LIBRARY
              CMAKE_CXX_MP_FLAG
              CMAKE_CXX_MP_NUM_PROCESSORS
              PROJECT_LLVM_INSTALL_PREFIX
              CMAKE_OSX_ARCHITECTURES
              CMAKE_OSX_DEPLOYMENT_TARGET
              CMAKE_OSX_SYSROOT
              CMAKE_CUDA_COMPILER
              CMAKE_CUDA_HOST_COMPILER
              CMAKE_CUDA_STANDARD
              CMAKE_CUDA_ARCHITECTURES
              CMAKE_CUDA_FLAGS
              CMAKE_CUDA_RUNTIME_LIBRARY
              CUDAToolkit_ROOT
              LINALG_CBLAS_INCLUDE_DIR
              LINALG_LAPACKE_INCLUDE_DIR
              LINALG_LAPACKE_LIBRARY
              LINALG_BLIS_LAPACK_LIBRARY
              LINALG_BLAS_RUNTIME_LIBRARY
              LINALG_LAPACKE_RUNTIME_LIBRARY
              BLA_VENDOR
              BLA_STATIC
              BLA_SIZEOF_INTEGER
              BLA_THREAD
              BLA_PREFER_PKGCONFIG
              MKL_ROOT
              MKL_INTERFACE
              MKL_LINK
              MKL_THREADING
              LINALG_CCACHE_PROGRAM
              LINALG_SCCACHE_PROGRAM
              LINALG_BUILDCACHE_PROGRAM
              LINALG_ICECC_EXECUTABLE
              CLANG_TIDY_PATH
              IWYU_EXECUTABLE
              CODESPELL_EXECUTABLE
              PROJECT_VALGRIND_TIMEOUT_MULTIPLIER
              DART_TESTING_TIMEOUT
              MEMORYCHECK_COMMAND
              MEMORYCHECK_COMMAND_OPTIONS
              MEMORYCHECK_SUPPRESSIONS_FILE
              CMAKE_PDB_OUTPUT_DIRECTORY
    )
        if(DEFINED ${_summary_var})
            message("  ${_summary_var} : ${${_summary_var}}")
        endif()
    endforeach()
    # This environment switch controls automatic BLAS prefix discovery.
    if(DEFINED ENV{LINALG_NO_AUTO_BLAS_PREFIX})
        message("  LINALG_NO_AUTO_BLAS_PREFIX (env): $ENV{LINALG_NO_AUTO_BLAS_PREFIX}")
    endif()
    message("")
    unset(_summary_configs)
    unset(_summary_config)
    unset(_summary_config_upper)
    unset(_summary_flag)
    unset(_summary_var)
    unset(_summary_compile_defs)
endif()
