# Runs even on CPU-only CI: prevent implicit host transfers from returning.
file(GLOB_RECURSE gpu_sources "${SOURCE_ROOT}/src/matrix_operation/gpu/*.cpp"
                              "${SOURCE_ROOT}/src/matrix_operation/gpu/*.cu")
if(NOT gpu_sources)
    message(FATAL_ERROR "No GPU sources found")
endif()
foreach(source IN LISTS gpu_sources)
    file(READ "${source}" contents)
    # Device copies are allowed; host copies, implicit default-direction copies,
    # managed-memory migration and host result accessors are not.
    if(contents MATCHES "cudaMemcpy(HostToDevice|DeviceToHost|Default)|cuda(Device|Stream)Synchronize|cudaMallocManaged|cudaMemPrefetchAsync|cublas(Get|Set)(Vector|Matrix)")
        message(FATAL_ERROR "GPU residency violation in ${source}")
    endif()
    string(REGEX MATCHALL "cudaMemcpy[A-Za-z0-9_]*[ \t\r\n]*\\([^;]*" copies "${contents}")
    foreach(copy IN LISTS copies)
        if(NOT copy MATCHES "cudaMemcpyDeviceToDevice")
            message(FATAL_ERROR "Unclassified GPU copy in ${source}: ${copy}")
        endif()
    endforeach()
endforeach()
# CUDA kernels and library bridges must stay in the GPU implementation tree.
file(GLOB_RECURSE cpu_sources "${SOURCE_ROOT}/src/matrix_operation/cpu/*.cpp"
                              "${SOURCE_ROOT}/src/matrix_operation/cpu/*.cu")
foreach(source IN LISTS cpu_sources)
    file(READ "${source}" contents)
    if(contents MATCHES "__global__|cudaMemcpy|cublas_v2.h|cusolverDn.h|cuda_runtime")
        message(FATAL_ERROR "GPU implementation outside GPU directory: ${source}")
    endif()
endforeach()
