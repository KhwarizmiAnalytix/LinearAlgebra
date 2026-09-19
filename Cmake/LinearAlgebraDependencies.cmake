# =============================================================================
# LinearAlgebra — test dependency resolution (standalone + embedded)
# =============================================================================
# Resolves Google Test without requiring an XSigma tree. MKL (optional,
# LINALG_ENABLE_MKL) is resolved directly in the top-level CMakeLists.txt via
# find_package(MKL), since it is a library dependency, not a test dependency.
# =============================================================================

include_guard(GLOBAL)

include(third_party_helpers)

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
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        add_subdirectory("${LINALG_THIRD_PARTY_DIR}/googletest"
                         "${CMAKE_BINARY_DIR}/ThirdParty/googletest_build" EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.18.0
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    # XSigma-style namespaced aliases used by this repo's test targets.
    _create_third_party_interface_targets(
        "Gtest::gtest=GTest::gtest|gtest"
        "Gtest::gtest_main=GTest::gtest_main|gtest_main"
    )
endfunction()
