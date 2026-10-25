load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for LinearAlgebra — mirrors CMake LINALG_CXX_STANDARD (default: 20)
LINALG_CXX_STD = "c++20"

def linear_algebra_copts():
    return xsigma_copts(cxx_std = LINALG_CXX_STD)

def linear_algebra_defines():
    """Returns compile definitions for linear_algebra.

    Mirrors CMakeLists.txt: LINALG_HAS_MKL / LINALG_HAS_BLAS /
    LINALG_HAS_LU_PIVOTING, toggled by the linalg_enable_mkl, linalg_enable_blas,
    linalg_lu_pivoting --define flags. These are always-defined 0/1 macros
    (checked in C++ with plain #if, never #ifdef/#if defined), matching
    CMakeLists.txt's target_compile_definitions($<BOOL:...>) scheme, not
    conditionally-defined ones. Like linalg_enable_mkl, linalg_enable_blas
    only threads the compile define through — it does not resolve cblas.h/
    lapacke.h or link a BLAS/LAPACK library the way CMakeLists.txt's
    find_package(BLAS)/find_path() do, so a build with it set still depends
    on those headers/libs already being on the toolchain's default search
    paths. CMake remains this repo's primary, exercised build system for the
    blas_lapack backend (see BUILD.bazel). LINALG_HAS_LAPACKE,
    LINALG_HAS_ACCELERATE, LINALG_HAS_CUBLAS, and
    LINALG_HAS_WIN32_WARNINGS have no Bazel toggle at all (see BUILD.bazel);
    they are simply left undefined here, which the #if sites that reference
    them treat as 0 — the same behavior as before this macro rename.
    """
    defines = xsigma_defines()

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

    # Bazel preserves the scalar algorithm configuration; CMake selects SIMD.
    defines += ["LINALG_HAS_VECTORIZATION=0"]
    return defines

def linear_algebra_linkopts():
    return xsigma_linkopts()
