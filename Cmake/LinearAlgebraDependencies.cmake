# =============================================================================
# LinearAlgebra — test dependency resolution (standalone + embedded)
# =============================================================================
# Resolves Google Test and (optionally) Google Benchmark without requiring an
# XSigma tree. MKL/BLAS/cuBLAS/cpuinfo (optional, LINALG_ENABLE_MKL/BLAS/
# CUBLAS/CPUINFO) are resolved directly in the top-level CMakeLists.txt via
# find_package(), since they are library dependencies, not test dependencies.
# =============================================================================

include_guard(GLOBAL)

include("${LINALG_SOURCE_DIR}/Cmake/third_party_helpers.cmake")

# Keep every bundled target on the compiler's default macOS C++ runtime. The
# bundled Logging helper injects Homebrew LLVM search paths even for AppleClang.
# Remove only that helper's directory options at this integration boundary.
function(linalg_remove_dependency_runtime_options directory)
    get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_options "${_target}" LINK_OPTIONS)
        if(_options)
            list(REMOVE_ITEM _options ${ARGN})
            set_property(TARGET "${_target}" PROPERTY LINK_OPTIONS "${_options}")
        endif()
    endforeach()
    get_property(_children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(_child IN LISTS _children)
        linalg_remove_dependency_runtime_options("${_child}" ${ARGN})
    endforeach()
endfunction()

# MSVC also identifies clang++ targeting the MSVC ABI. These bundled projects
# use it to select cl.exe arguments, which the GNU Clang driver cannot parse.
# Limit the override to dependency configuration; keep the project's ABI and
# CUDA host compiler settings intact in the caller.
function(linalg_add_dependency_subdirectory source_dir binary_dir)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                            STREQUAL "GNU"
    )
        set(MSVC FALSE)
    endif()
    get_property(
        _had_testing
        CACHE BUILD_TESTING
        PROPERTY TYPE
        SET
    )
    if(_had_testing)
        get_property(_testing_value CACHE BUILD_TESTING PROPERTY VALUE)
        get_property(_testing_help CACHE BUILD_TESTING PROPERTY HELPSTRING)
        get_property(_testing_type CACHE BUILD_TESTING PROPERTY TYPE)
    endif()
    add_subdirectory("${source_dir}" "${binary_dir}" ${ARGN})
    if(APPLE)
        get_directory_property(
            _dependency_runtime_options DIRECTORY "${source_dir}" DEFINITION LLVM_LINK_FLAGS
        )
        if(_dependency_runtime_options)
            linalg_remove_dependency_runtime_options("${source_dir}" ${_dependency_runtime_options})
        endif()
    endif()
    if(_had_testing)
        set(BUILD_TESTING "${_testing_value}" CACHE ${_testing_type} "${_testing_help}" FORCE)
    else()
        unset(BUILD_TESTING CACHE)
    endif()
endfunction()

# CMAKE_CURRENT_LIST_DIR = this file's directory (<repo>/Cmake) regardless of
# the including directory's scope.
get_filename_component(_linalg_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LINALG_THIRD_PARTY_DIR "${_linalg_repo_root}/ThirdParty"
    CACHE PATH "Root of LinearAlgebra's bundled third-party sources"
)

# -----------------------------------------------------------------------------
# linalg_setup_gtest: Google Test for the test suite
# -----------------------------------------------------------------------------
function(linalg_setup_gtest)
    if(TARGET gtest_main OR TARGET GTest::gtest_main)
        # Fall through to alias normalization below.
    elseif(COMMAND xsigma_add_googletest)
        # Embedded in XSigma: reuse the host's googletest wiring.
        xsigma_add_googletest()
    elseif(EXISTS "${LINALG_THIRD_PARTY_DIR}/googletest/CMakeLists.txt")
        set(INSTALL_GTEST OFF)
        set(BUILD_GMOCK OFF)
        set(gtest_force_shared_crt ON)
        linalg_add_dependency_subdirectory(
            "${LINALG_THIRD_PARTY_DIR}/googletest"
            "${LINALG_BINARY_DIR}/ThirdParty/googletest_build" EXCLUDE_FROM_ALL
        )
    else()
        include(FetchContent)
        FetchContent_Declare(
            googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.18.0
        )
        set(INSTALL_GTEST OFF)
        set(BUILD_GMOCK OFF)
        set(gtest_force_shared_crt ON)
        FetchContent_MakeAvailable(googletest)
    endif()
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    # XSigma-style namespaced aliases used by this repo's test targets.
    linalg_create_third_party_interface_targets(
        "Gtest::gtest=GTest::gtest|gtest" "Gtest::gtest_main=GTest::gtest_main|gtest_main"
    )
endfunction()

# -----------------------------------------------------------------------------
# linalg_setup_benchmark: Google Benchmark for micro-benchmarks
# -----------------------------------------------------------------------------
function(linalg_setup_benchmark)
    if(TARGET benchmark OR TARGET benchmark::benchmark)
        return()
    endif()
    # Prefer the vendored submodule (pinned commit, known-good): some system
    # benchmark packages (e.g. static Homebrew archives) mislink/crash when
    # consumed via an IMPORTED target with a different toolchain.
    set(_bench_src "")
    if(FETCHCONTENT_SOURCE_DIR_BENCHMARK)
        # An explicit FetchContent source override takes precedence over bundled copies.
    elseif(EXISTS "${LINALG_THIRD_PARTY_DIR}/benchmark/CMakeLists.txt")
        set(_bench_src "${LINALG_THIRD_PARTY_DIR}/benchmark")
    elseif(EXISTS "${_linalg_repo_root}/ThirdParty/Logging/ThirdParty/benchmark/CMakeLists.txt")
        set(_bench_src "${_linalg_repo_root}/ThirdParty/Logging/ThirdParty/benchmark")
    elseif(LINALG_ENABLE_EXTERNAL)
        find_package(benchmark QUIET)
        if(benchmark_FOUND)
            message(STATUS "Found external Google Benchmark")
            return()
        endif()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-c2y-extensions")
    endif()
    if(NOT DEFINED HAVE_STD_REGEX)
        set(HAVE_STD_REGEX 1 CACHE INTERNAL "")
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "Enable benchmark exceptions" FORCE)
    set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "Disable benchmark LTO" FORCE)
    set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "Disable benchmark libcxx" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable benchmark werror" FORCE)
    set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "Disable benchmark force werror" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs install" FORCE)
    set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE BOOL "Disable benchmark doxygen" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests" FORCE)
    set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest for benchmark" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Don't download dependencies" FORCE)
    if(_bench_src)
        linalg_add_dependency_subdirectory(
            "${_bench_src}" "${LINALG_BINARY_DIR}/ThirdParty/benchmark_build" EXCLUDE_FROM_ALL
        )
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                                STREQUAL "GNU"
        )
            # The GNU driver treats a bare shlwapi.lib as an input file, rather
            # than searching the Windows SDK as the MSVC driver does.
            get_target_property(_bench_links benchmark LINK_LIBRARIES)
            list(TRANSFORM _bench_links REPLACE "^shlwapi$" "-lshlwapi")
            set_property(TARGET benchmark PROPERTY LINK_LIBRARIES "${_bench_links}")
        endif()
    else()
        include(FetchContent)
        FetchContent_Declare(
            benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.9.5
        )
        FetchContent_MakeAvailable(benchmark)
    endif()
    if(TARGET benchmark)
        get_target_property(_benchmark_binary_dir benchmark BINARY_DIR)
        linalg_set_third_party_folder_properties("benchmark" "${_benchmark_binary_dir}")
    endif()
endfunction()

# Always provide the bundled reference implementation. It remains private to
# validation targets, so installed library consumers do not need Eigen headers.
function(linalg_setup_eigen)
    if(TARGET Eigen3::Eigen)
        return()
    endif()
    set(_eigen_source "${LINALG_THIRD_PARTY_DIR}/eigen")
    if(NOT EXISTS "${_eigen_source}/Eigen/Core")
        message(
            FATAL_ERROR
                "Bundled Eigen headers are missing. Run git submodule update --init --recursive."
        )
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_eigen_source}")
endfunction()
