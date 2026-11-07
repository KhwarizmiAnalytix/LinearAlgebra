# =============================================================================
# cuda.cmake — CUDA toolchain, architecture detection, and per-target flags
# for LinearAlgebra.
#
# Included by CMakeLists.txt when LINALG_GPU_BACKEND=cuda. After inclusion:
#   LINALG_CUDA_ARCHITECTURES   — resolved SM list (e.g. "75;80;86;90") or "native"
#   LINALG_CUDA_HOST_COMPILER   — optional host compiler forwarded to -ccbin
#
# Callers must include before add_library() so that CMAKE_CUDA_ARCHITECTURES
# and CMAKE_CUDA_STANDARD are already set when enable_language(CUDA) configures
# the NVCC toolchain.
#
# Function exported after inclusion:
#   linalg_apply_cuda_flags(<target>)
#       Sets CUDA_STANDARD / CUDA_ARCHITECTURES / CUDA_SEPARABLE_COMPILATION
#       and per-compiler nvcc / clang-CUDA compile options on <target>.
#       Call after add_library().
#
# Design follows PyTorch's cmake/Cuda.cmake (TORCH_CUDA_ARCH_LIST convention
# and -ccbin host-compiler forwarding) and Scalars' cmake/gpu_backend.cmake
# (architecture cache var, MSVC preprocessor workaround, clang fast-math).
# =============================================================================
include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# find_package(CUDAToolkit) — provides CUDA::cudart, CUDA::cublas,
# CUDA::cusolver, and CUDAToolkit_LIBRARY_DIR used by the WIN32/GNU link fix.
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# LINALG_CUDA_ARCHITECTURES
#
# Mirrors PyTorch's TORCH_CUDA_ARCH_LIST: a semicolon-separated SM list
# (e.g. "75;80;86;89;90") or the special value "native" which asks nvcc to
# target the GPU present on the build machine at compile time. Falls through
# to CMAKE_CUDA_ARCHITECTURES when the caller already set it, so the standard
# CMake override path (-DCMAKE_CUDA_ARCHITECTURES=...) still works without
# the LINALG_-prefixed variable. The default list (75;80;86;89;90) covers
# Turing / Ampere / Ada Lovelace — Scalars uses the same default for
# SCALARS_CUDA_ARCHITECTURES. Use "native" for single-GPU developer builds.
# ---------------------------------------------------------------------------
if(NOT DEFINED LINALG_CUDA_ARCHITECTURES)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
    )
        set(LINALG_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CACHE STRING
                  "CUDA compute capability list for LinearAlgebra (e.g. 75;80;86;90 or native)"
        )
    else()
        set(LINALG_CUDA_ARCHITECTURES "native"
            CACHE STRING
                  "CUDA compute capability list for LinearAlgebra (e.g. 75;80;86;90 or native)"
        )
    endif()
endif()

# Propagate the resolved value back to CMAKE_CUDA_ARCHITECTURES so that
# enable_language(CUDA) picks it up and the nvcc toolchain detection uses it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${LINALG_CUDA_ARCHITECTURES}")
endif()

# ---------------------------------------------------------------------------
# LINALG_CUDA_HOST_COMPILER
#
# Optional: forward a specific host C++ compiler to nvcc via -ccbin, matching
# PyTorch's -DCMAKE_CUDA_HOST_COMPILER=... idiom. Many CI configurations
# set this to avoid NVCC picking up a mismatched g++ when the primary CXX
# compiler is clang++. Empty string = let NVCC choose its own host compiler.
# ---------------------------------------------------------------------------
set(LINALG_CUDA_HOST_COMPILER ""
    CACHE FILEPATH
          "Host C++ compiler forwarded to nvcc as -ccbin (empty = nvcc picks its own)"
)
mark_as_advanced(LINALG_CUDA_HOST_COMPILER)
if(LINALG_CUDA_HOST_COMPILER AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${LINALG_CUDA_HOST_COMPILER}" CACHE FILEPATH "" FORCE)
endif()

# ---------------------------------------------------------------------------
# CMAKE_CUDA_STANDARD — inherit from the project's CXX standard. Must be set
# before enable_language(CUDA) so the NVCC toolchain picks it up during its
# first-run detection; the linalg_apply_cuda_flags() function re-applies it
# as a per-target CUDA_STANDARD property for clarity.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${LINALG_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ---------------------------------------------------------------------------
# WIN32 / clang GNU-driver link-options workaround.
#
# Mixed CXX+CUDA targets link with clang++. CMake's CUDA runtime linker
# options for shared/static libraries are bare names (cudadevrt, cudart,
# cudart_static), which the MSVC linker frontend expands to *.lib search
# entries. The GNU driver treats those bare names as input files — producing
# "no such file" errors — instead of searching the CUDA library directory.
# Override with full absolute paths after enable_language(CUDA) loads the
# compiler module and its default runtime-library link options.
#
# Same workaround applied in Scalars' cmake/gpu_backend.cmake (CUDA block).
# ---------------------------------------------------------------------------
enable_language(CUDA)

if(WIN32
   AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
)
    set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED
        "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
        "${CUDAToolkit_LIBRARY_DIR}/cudart.lib"
    )
    set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC
        "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
        "${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib"
    )
endif()


# ---------------------------------------------------------------------------
# linalg_apply_cuda_flags(<target>)
#
# Sets CUDA_STANDARD, CUDA_ARCHITECTURES, and CUDA_SEPARABLE_COMPILATION as
# target properties, then applies per-compiler compile options:
#   NVIDIA nvcc  — --expt-relaxed-constexpr (Scalars operators are constexpr;
#                  device kernels call them without host/device dispatch macros)
#   MSVC host    — /Zc:preprocessor for both CXX and CUDA TUs
#                  (CUDA 13.x CCCL requires the conforming preprocessor;
#                  forwarded via -Xcompiler for .cu files whose host pass uses cl.exe)
#
# Design parallels scalars_apply_gpu_backend() in Scalars' cmake/gpu_backend.cmake
# and PyTorch's torch_set_target_properties() in cmake/TorchConfig.cmake.in.
# ---------------------------------------------------------------------------
function(linalg_apply_cuda_flags target)
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_STANDARD ${LINALG_CXX_STANDARD}
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${LINALG_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION OFF
    )

    target_compile_options(
        ${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
    )

    if(MSVC)
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()
endfunction()
