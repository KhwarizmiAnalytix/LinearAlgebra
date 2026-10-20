# =============================================================================
# LinearAlgebra Library C++
# Micro-benchmarks
# =============================================================================
include(LinearAlgebraDependencies)

linalg_setup_benchmark()
_create_third_party_interface_targets(
    "Benchmark::benchmark=benchmark::benchmark|benchmark"
    "Benchmark::benchmark_main=benchmark::benchmark_main|benchmark_main"
)
if(NOT TARGET Benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark target unavailable after linalg_setup_benchmark()")
endif()

file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cxx")

foreach(_bench_source IN ITEMS ${bench_sources})
    get_filename_component(bench_name "${_bench_source}" NAME_WE)
    string(REPLACE "Benchmark" "" bench_suffix "${bench_name}")
    string(TOLOWER "${bench_suffix}" bench_suffix_lower)
    set(target_name "benchmark_linalg_${bench_suffix_lower}")

    add_executable(${target_name} "${_bench_source}")
    target_compile_definitions(
        ${target_name} PRIVATE "LINALG_BENCHMARK_BLIS=$<BOOL:${LINALG_ENABLE_BLIS}>"
    )
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

    add_test(NAME ${target_name} COMMAND ${target_name} --benchmark_min_time=0.01s
             WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
    set_tests_properties(
        ${target_name}
        PROPERTIES TIMEOUT
                   600
                   LABELS
                   "Benchmark;Performance;LinearAlgebra"
                   PROCESSOR_AFFINITY
                   TRUE
                   PROCESSORS
                   1
                   ENVIRONMENT
                   "MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    )
endforeach()
