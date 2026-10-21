# =============================================================================
# LinearAlgebra Library C++
# Tests
# =============================================================================
include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")

if(LINALG_ENABLE_GTEST)
    linalg_setup_gtest()

    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx"
    )

    add_executable(LinearAlgebraCxxTests ${test_sources})

    # Testing/ (sources + generated *Test.h) and repo-root includes match the
    # LinearAlgebra::LinearAlgebra interface.
    target_include_directories(
        LinearAlgebraCxxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.."
                                      "${CMAKE_CURRENT_BINARY_DIR}/.."
    )

    target_compile_definitions(
        LinearAlgebraCxxTests PRIVATE LINALG_HAS_GTEST=1 LINALG_GOOGLE_TEST USE_GTEST
                                      _VARIADIC_MAX=10
    )

    target_link_libraries(
        LinearAlgebraCxxTests PRIVATE Gtest::gtest Gtest::gtest_main LinearAlgebra::LinearAlgebra
    )

    set_target_properties(
        LinearAlgebraCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                                        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    add_test(NAME LinearAlgebraCxxTests COMMAND LinearAlgebraCxxTests --gtest_filter=-MathGpu.*
             WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    set_tests_properties(LinearAlgebraCxxTests PROPERTIES TIMEOUT 120 LABELS "LinearAlgebra;Unit")

    if(LINALG_ENABLE_CUBLAS)
        add_test(NAME LinearAlgebraCudaTests COMMAND LinearAlgebraCxxTests --gtest_filter=MathGpu.*
                 WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
        set_tests_properties(
            LinearAlgebraCudaTests PROPERTIES TIMEOUT 120 LABELS "LinearAlgebra;Unit;CUDA"
        )
    endif()

    # LINALG_ENABLE_CUBLAS and LINALG_ENABLE_ROCBLAS are mutually exclusive
    # (CMakeLists.txt), so this is the same MathGpu.* suite as
    # LinearAlgebraCudaTests above, compiled against ROCM_BLAS instead.
    if(LINALG_ENABLE_ROCBLAS)
        add_test(NAME LinearAlgebraRocmTests COMMAND LinearAlgebraCxxTests --gtest_filter=MathGpu.*
                 WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
        set_tests_properties(
            LinearAlgebraRocmTests PROPERTIES TIMEOUT 120 LABELS "LinearAlgebra;Unit;ROCM"
        )
    endif()

    if(COMMAND linalg_set_target_output_directories)
        linalg_set_target_output_directories(LinearAlgebraCxxTests)
    endif()

endif()

add_test(NAME GpuResidencySourceCheck
         COMMAND ${CMAKE_COMMAND} -DSOURCE_ROOT=${LINALG_SOURCE_DIR} -P
                 ${CMAKE_CURRENT_SOURCE_DIR}/CheckGpuResidency.cmake
)
set_tests_properties(GpuResidencySourceCheck PROPERTIES LABELS "LinearAlgebra;Unit")
