# =============================================================================
# LinearAlgebra Library C++
# Tests
# =============================================================================
include(LinearAlgebraDependencies)

linalg_setup_gtest()

file(GLOB_RECURSE test_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx")

add_executable(LinearAlgebraCxxTests ${test_sources})

# Testing/ (sources + generated *Test.h) and repo-root includes match the
# LinearAlgebra::LinearAlgebra interface.
target_include_directories(
  LinearAlgebraCxxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
)

target_compile_definitions(
  LinearAlgebraCxxTests PRIVATE LINALG_HAS_GTEST=1 LINALG_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
)

target_link_libraries(
  LinearAlgebraCxxTests PRIVATE Gtest::gtest Gtest::gtest_main LinearAlgebra::LinearAlgebra
)

set_target_properties(
  LinearAlgebraCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_test(NAME LinearAlgebraCxxTests COMMAND LinearAlgebraCxxTests
         WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_tests_properties(LinearAlgebraCxxTests PROPERTIES TIMEOUT 120 LABELS "LinearAlgebra;Unit")

if(COMMAND linalg_set_target_output_directories)
  linalg_set_target_output_directories(LinearAlgebraCxxTests)
endif()
