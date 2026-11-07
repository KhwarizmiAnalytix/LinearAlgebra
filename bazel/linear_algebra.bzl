"""Compiler options and feature definitions for LinearAlgebra targets."""

# C++ standard for LinearAlgebra — mirrors CMake LINALG_CXX_STANDARD (default: 20)
LINALG_CXX_STD = "c++20"

def linear_algebra_copts():
    return select({
        "@platforms//os:windows": [
            "/std:" + LINALG_CXX_STD,
            "/Zc:__cplusplus",  # expose correct __cplusplus value (mirrors CMake /Zc:__cplusplus)
            "/EHsc",  # structured exception handling
            "/bigobj",  # large object files (mirrors CMake /bigobj)
            "/utf-8",  # UTF-8 source/output encoding (mirrors CMake /utf-8)
            "/wd4244",  # narrowing conversion (mirrors CMake /wd4244)
            "/wd4267",  # size_t → int conversion (mirrors CMake /wd4267)
            "/wd4715",  # not all control paths return (mirrors CMake /wd4715)
            "/wd4018",  # signed/unsigned comparison (mirrors CMake /wd4018)
            "/WX",  # warnings as errors (mirrors CMake /WX)
        ],
        "//conditions:default": [
            "-std=" + LINALG_CXX_STD,
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-include",
            "cstdlib",
        ],
    })

def linear_algebra_gpu_copts():
    """Returns extra compile options for GPU backend bridge sources.

    The CUDA SDK headers require the conforming preprocessor on Windows.
    SDK headers and libraries are provided by GPU target dependencies.
    """
    return select({
        Label("//bazel:linalg_gpu_cuda_windows"): ["/Zc:preprocessor"],
        "//conditions:default": [],
    })

def linear_algebra_arch_copts():
    """ISA options for Scalars and every translation unit using its headers."""
    return select({
        Label("//bazel:linalg_cpu_sse"): ["-msse4.2"],
        Label("//bazel:linalg_cpu_avx"): ["-mavx", "-mfma", "-mf16c"],
        Label("//bazel:linalg_cpu_avx2"): ["-mavx2", "-mfma", "-mf16c"],
        Label("//bazel:linalg_cpu_avx512"): ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512cd", "-mfma", "-mf16c"],
        Label("//bazel:linalg_cpu_sse_windows"): ["/arch:SSE2"],
        Label("//bazel:linalg_cpu_avx_windows"): ["/arch:AVX", "/D__F16C__"],
        Label("//bazel:linalg_cpu_avx2_windows"): ["/arch:AVX2", "/D__F16C__"],
        Label("//bazel:linalg_cpu_avx512_windows"): ["/arch:AVX512", "/D__F16C__"],
        Label("//bazel:linalg_cpu_neon"): ["-march=armv8-a+fp16"],
        Label("//bazel:linalg_cpu_sve"): ["-march=armv8-a+sve", "-msve-vector-bits=128"],
        "//conditions:default": [],
    })

def linear_algebra_defines():
    """Returns compile definitions for linear_algebra.

    Mirrors CMakeLists.txt: LINALG_HAS_MKL / LINALG_HAS_BLAS /
    LINALG_HAS_LU_PIVOTING / LINALG_HAS_CUBLAS / LINALG_HAS_ROCBLAS, toggled
    by the matching --define flags. These are always-defined 0/1 macros
    (checked in C++ with plain #if, never #ifdef/#if defined), matching
    CMakeLists.txt's target_compile_definitions($<BOOL:...>) scheme.

    LINALG_HAS_CUBLAS / LINALG_HAS_ROCBLAS: the GPU backend bridge .cpp
    sources and .cu/.hip kernel files are selected in BUILD.bazel; only the
    bridge .cpp sources compile with the default cc toolchain. The .cu kernel
    files are compiled by :linear_algebra_cuda_kernels (rules_cuda) or
    :linear_algebra_hip_kernels (ROCm Clang). Their SDK dependencies supply
    the CUDA/HIP library headers and runtime libraries.

    LINALG_HAS_VECTORIZATION: controlled by --define=linalg_cpu_backend=<isa>.
    The corresponding SIMD compiler flags (-mavx2, -msse4.1, etc.) are applied
    in bazel/scalars.BUILD and forwarded to Scalars::Scalars so that the
    Scalars SIMD backend and LinearAlgebra agree on the active ISA.

    LINALG_HAS_LAPACKE, LINALG_HAS_ACCELERATE have no Bazel toggles; the #if
    sites treat them as 0. LINALG_HAS_WIN32_WARNINGS is toggled by
    --define=linalg_display_win32_warnings=true (default 0, mirrors CMake's
    LINALG_DISPLAY_WIN32_WARNINGS default OFF).

    Returns:
        Configurable feature definitions for the library and its consumers.
    """

    # Windows CRT safety defines (mirrors CMake compiler_checks.cmake).
    defines = select({
        "@platforms//os:windows": [
            "_CRT_SECURE_NO_DEPRECATE",
            "_CRT_NONSTDC_NO_DEPRECATE",
            "_CRT_SECURE_NO_WARNINGS",
            "_SCL_SECURE_NO_DEPRECATE",
            "_SCL_SECURE_NO_WARNINGS",
        ],
        "//conditions:default": [],
    })

    defines += select({
        "//bazel:linalg_enable_mkl": ["LINALG_HAS_MKL=1"],
        "//conditions:default": ["LINALG_HAS_MKL=0"],
    })
    defines += select({
        "//bazel:linalg_enable_blas": ["LINALG_HAS_BLAS=1"],
        "//conditions:default": ["LINALG_HAS_BLAS=0"],
    })
    defines += select({
        "//bazel:linalg_lu_pivoting": ["LINALG_HAS_LU_PIVOTING=1"],
        "//conditions:default": ["LINALG_HAS_LU_PIVOTING=0"],
    })
    defines += select({
        "//bazel:linalg_win32_warnings": ["LINALG_HAS_WIN32_WARNINGS=1"],
        "//conditions:default": ["LINALG_HAS_WIN32_WARNINGS=0"],
    })

    # GPU backend defines — mirror CMakeLists.txt's $<STREQUAL:${LINALG_GPU_BACKEND},cuda/hip>.
    defines += select({
        "//bazel:linalg_gpu_cuda": ["LINALG_HAS_CUBLAS=1", "LINALG_HAS_ROCBLAS=0"],
        "//bazel:linalg_gpu_hip": ["LINALG_HAS_CUBLAS=0", "LINALG_HAS_ROCBLAS=1"],
        "//conditions:default": ["LINALG_HAS_CUBLAS=0", "LINALG_HAS_ROCBLAS=0"],
    })

    # CPU SIMD backend — LINALG_HAS_VECTORIZATION mirrors CMakeLists.txt's
    # $<NOT:$<STREQUAL:${LINALG_CPU_BACKEND},no>>. The default (no --define)
    # is treated as scalar (0) to preserve backwards compatibility; pass
    # --define=linalg_cpu_backend=avx2 (or neon / sse / ...) to enable.
    defines += select({
        "//bazel:linalg_cpu_no": ["LINALG_HAS_VECTORIZATION=0"],
        "//bazel:linalg_cpu_sse": ["LINALG_HAS_VECTORIZATION=1"],
        "//bazel:linalg_cpu_avx": ["LINALG_HAS_VECTORIZATION=1"],
        "//bazel:linalg_cpu_avx2": ["LINALG_HAS_VECTORIZATION=1"],
        "//bazel:linalg_cpu_avx512": ["LINALG_HAS_VECTORIZATION=1"],
        "//bazel:linalg_cpu_neon": ["LINALG_HAS_VECTORIZATION=1"],
        "//bazel:linalg_cpu_sve": ["LINALG_HAS_VECTORIZATION=1"],
        "//conditions:default": ["LINALG_HAS_VECTORIZATION=0"],
    })
    return defines

def linear_algebra_linkopts():
    return select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [
            "-undefined",
            "dynamic_lookup",
        ],
        "//conditions:default": [
            "-lpthread",
            "-ldl",
        ],
    })
