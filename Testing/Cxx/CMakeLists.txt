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

if(LINALG_ENABLE_BENCHMARK)
  linalg_setup_benchmark()
  _create_third_party_interface_targets(
    "Benchmark::benchmark=benchmark::benchmark|benchmark"
    "Benchmark::benchmark_main=benchmark::benchmark_main|benchmark_main"
  )
  if(NOT TARGET Benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark target unavailable after linalg_setup_benchmark()")
  endif()

  file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false
       "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cxx"
  )

  foreach(_bench_source IN ITEMS ${bench_sources})
    get_filename_component(bench_name "${_bench_source}" NAME_WE)
    string(REPLACE "Benchmark" "" bench_suffix "${bench_name}")
    string(TOLOWER "${bench_suffix}" bench_suffix_lower)
    set(target_name "benchmark_linalg_${bench_suffix_lower}")

    add_executable(${target_name} "${_bench_source}")
    target_include_directories(
      ${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    target_link_libraries(
      ${target_name} PRIVATE Benchmark::benchmark Benchmark::benchmark_main
                              LinearAlgebra::LinearAlgebra
    )
    if(COMMAND linalg_set_target_output_directories)
      linalg_set_target_output_directories(${target_name})
    endif()
    set_target_properties(${target_name} PROPERTIES FOLDER "Benchmarks")
  endforeach()
endif()
