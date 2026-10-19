# This example uses all four providers in the same executable.
if(NOT LINALG_ENABLE_BLAS
   OR NOT LINALG_BLAS_HAS_LAPACKE
   OR NOT LINALG_ENABLE_MKL
   OR NOT LINALG_ENABLE_CUBLAS
)
    message(
        FATAL_ERROR
            "The backend benchmark example requires BLAS with LAPACKE, MKL and cuBLAS enabled"
    )
endif()

set(_example_results "${CMAKE_CURRENT_BINARY_DIR}/results")
file(MAKE_DIRECTORY "${_example_results}")

add_executable(backend_analysis benchmark.cpp)
add_executable(transfer_analysis transfer.cpp)
foreach(_example IN ITEMS backend_analysis transfer_analysis)
    target_compile_features(${_example} PRIVATE cxx_std_17)
    target_link_libraries(${_example} PRIVATE LinearAlgebra::LinearAlgebra)
    set_target_properties(${_example} PROPERTIES FOLDER "Examples")
    if(COMMAND linalg_set_target_output_directories)
        linalg_set_target_output_directories(${_example})
    endif()
endforeach()

if(LINALG_ENABLE_TESTING)
    add_test(NAME example_backend_analysis COMMAND backend_analysis
                                                   "${_example_results}/results.csv"
    )
    add_test(NAME example_backend_analysis_rerun
             COMMAND backend_analysis "${_example_results}/rerun_results.csv"
                     "${CMAKE_CURRENT_SOURCE_DIR}/selections/rerun_selection.txt"
    )
    add_test(NAME example_backend_analysis_gpu_steady
             COMMAND backend_analysis "${_example_results}/gpu_steady_results.csv"
                     "${CMAKE_CURRENT_SOURCE_DIR}/selections/gpu_steady_selection.txt" steady_gpu
    )
    add_test(NAME example_transfer_analysis COMMAND transfer_analysis
                                                    "${_example_results}/transfers.csv"
    )
    set_tests_properties(
        example_backend_analysis example_backend_analysis_rerun example_backend_analysis_gpu_steady
        example_transfer_analysis
        PROPERTIES PROCESSOR_AFFINITY
                   TRUE
                   PROCESSORS
                   1
                   TIMEOUT
                   1800
                   ENVIRONMENT
                   "MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                   LABELS
                   "Example;Benchmark;Performance"
    )
endif()
