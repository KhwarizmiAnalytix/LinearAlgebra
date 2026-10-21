cmake_minimum_required(VERSION 3.24)

# Runs even on CPU-only CI: prevent implicit host transfers from returning.
file(GLOB_RECURSE gpu_sources "${SOURCE_ROOT}/src/matrix_operation/gpu/*.cpp"
                              "${SOURCE_ROOT}/src/matrix_operation/gpu/*.cu"
                              "${SOURCE_ROOT}/src/matrix_operation/gpu/*.hip")
if(NOT gpu_sources)
    message(FATAL_ERROR "No GPU sources found")
endif()
# rank/condition_number, determinant, norm, trace, symmetric_eigen_backend
# and lu_decomposition intentionally synchronize their bound stream and
# return their scalar/bool result by value, matching the primary
# (Backend-templated) template's signature exactly instead of writing
# through a device output pointer (see docs/gpu_backend_design.md's "Device
# result API"). lu_decomposition additionally round-trips its pivot through
# the host to widen cuSOLVER/hipSOLVER's 32-bit ipiv into the caller's
# linalg_int (64-bit) device buffer. Every other GPU entry point stays fully
# device-resident and async, so this check still applies to them; these
# implementation files are exempt. The exemption is file-granular (this
# check has no per-function scoping), so note that inversion_{cublas,hip}
# also hosts the still-async `inversion<T, Backend>` specializations
# alongside `determinant<T, Backend>`.
set(residency_exempt_sources
    "rank_cublas.cu" "rank_hip.hip"
    "inversion_cublas.cu" "inversion_hip.hip"
    "norm_cublas.cu" "norm_hip.hip"
    "trace_cublas.cu" "trace_hip.hip"
    "eigen_cublas.cpp" "eigen_hip.cpp"
    "lu_decomposition_cublas.cpp" "lu_decomposition_hip.cpp")
foreach(source IN LISTS gpu_sources)
    get_filename_component(source_name "${source}" NAME)
    if(source_name IN_LIST residency_exempt_sources)
        continue()
    endif()
    file(READ "${source}" contents)
    # Device copies are allowed; host copies, implicit default-direction copies,
    # managed-memory migration and host result accessors are not. Checked for
    # both CUDA and HIP spellings (hipBLAS/hipSOLVER mirror cuBLAS/cuSOLVER's
    # API names 1:1).
    if(contents MATCHES "(cuda|hip)Memcpy(HostToDevice|DeviceToHost|Default)|(cuda|hip)(Device|Stream)Synchronize|(cuda|hip)(MallocManaged|HostMalloc)|(cuda|hip)MemPrefetchAsync|(cublas|hipblas)(Get|Set)(Vector|Matrix)")
        message(FATAL_ERROR "GPU residency violation in ${source}")
    endif()
    string(REGEX MATCHALL "(cuda|hip)Memcpy[A-Za-z0-9_]*[ \t\r\n]*\\([^;]*" copies "${contents}")
    foreach(copy IN LISTS copies)
        if(NOT copy MATCHES "(cuda|hip)MemcpyDeviceToDevice")
            message(FATAL_ERROR "Unclassified GPU copy in ${source}: ${copy}")
        endif()
    endforeach()
endforeach()
# CUDA/HIP kernels and library bridges must stay in the GPU implementation tree.
file(GLOB_RECURSE cpu_sources "${SOURCE_ROOT}/src/matrix_operation/cpu/*.cpp"
                              "${SOURCE_ROOT}/src/matrix_operation/cpu/*.cu"
                              "${SOURCE_ROOT}/src/matrix_operation/cpu/*.hip")
foreach(source IN LISTS cpu_sources)
    file(READ "${source}" contents)
    if(contents MATCHES "__global__|cudaMemcpy|cublas_v2.h|cusolverDn.h|cuda_runtime|hipMemcpy|hipblas.h|hipsolver.h|hip_runtime")
        message(FATAL_ERROR "GPU implementation outside GPU directory: ${source}")
    endif()
endforeach()
