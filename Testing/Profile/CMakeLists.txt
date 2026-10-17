# =============================================================================
# LinearAlgebra — Testing/Profile
# Profile every matrix operation (CPU native / blas_lapacke / mkl and GPU)
# using KhwarizmiAnalytix/Profiler instrumentation.
# Requires Profiler::Profiler (auto-detected in the top-level CMakeLists.txt
# via find_package(Profiler CONFIG QUIET); the directory is only added when
# LINALG_ENABLE_PROFILING is ON and LINALG_HAS_PROFILER is set to ON).
# =============================================================================

if(NOT TARGET Profiler::Profiler)
    message(
        FATAL_ERROR
            "Testing/Profile requires Profiler::Profiler but the target is not available. "
            "Build and install KhwarizmiAnalytix/Profiler and set CMAKE_PREFIX_PATH to its "
            "install prefix, then re-configure with -DLINALG_ENABLE_PROFILING=ON."
    )
endif()

add_executable(profile_linalg_matrix_operations ProfileMatrixOperations.cxx)

target_include_directories(
    profile_linalg_matrix_operations
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
)

target_link_libraries(
    profile_linalg_matrix_operations
    PRIVATE LinearAlgebra::LinearAlgebra Profiler::Profiler
)

set_target_properties(
    profile_linalg_matrix_operations
    PROPERTIES FOLDER "Profile" CXX_STANDARD ${LINALG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

if(COMMAND linalg_set_target_output_directories)
    linalg_set_target_output_directories(profile_linalg_matrix_operations)
endif()

add_test(
    NAME profile_linalg_matrix_operations
    COMMAND profile_linalg_matrix_operations 64
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)
set_tests_properties(
    profile_linalg_matrix_operations
    PROPERTIES TIMEOUT 600 LABELS "Profile;Performance;LinearAlgebra"
)

# ---------------------------------------------------------------------------
# Focused eigen profiler
# ---------------------------------------------------------------------------

add_executable(profile_linalg_eigen ProfileEigen.cxx)

target_include_directories(
    profile_linalg_eigen
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
)

target_link_libraries(
    profile_linalg_eigen
    PRIVATE LinearAlgebra::LinearAlgebra Profiler::Profiler
)

set_target_properties(
    profile_linalg_eigen
    PROPERTIES FOLDER "Profile" CXX_STANDARD ${LINALG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

if(COMMAND linalg_set_target_output_directories)
    linalg_set_target_output_directories(profile_linalg_eigen)
endif()

add_test(
    NAME profile_linalg_eigen
    COMMAND profile_linalg_eigen 64
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)
set_tests_properties(
    profile_linalg_eigen
    PROPERTIES TIMEOUT 300 LABELS "Profile;Performance;LinearAlgebra"
)
