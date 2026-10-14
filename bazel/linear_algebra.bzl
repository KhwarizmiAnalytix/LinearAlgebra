load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for LinearAlgebra — mirrors CMake LINALG_CXX_STANDARD (default: 17)
LINALG_CXX_STD = "c++17"

def linear_algebra_copts():
    return xsigma_copts(cxx_std = LINALG_CXX_STD)

def linear_algebra_defines():
    """Returns compile definitions for linear_algebra.

    Mirrors CMakeLists.txt: LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS /
    LINALG_LU_PIVOTING toggles. Like linalg_enable_mkl, linalg_enable_blas
    only threads the compile define through — it does not resolve cblas.h/
    lapacke.h or link a BLAS/LAPACK library the way CMakeLists.txt's
    find_package(BLAS)/find_path() do, so a build with it set still depends
    on those headers/libs already being on the toolchain's default search
    paths. CMake remains this repo's primary, exercised build system for the
    blas_lapack backend (see BUILD.bazel).
    """
    defines = xsigma_defines()

    defines += select({
        "//bazel:linalg_enable_mkl": ["LINALG_ENABLE_MKL=1"],
        "//conditions:default": [],
    })
    defines += select({
        "//bazel:linalg_enable_blas": ["LINALG_ENABLE_BLAS=1"],
        "//conditions:default": [],
    })
    defines += select({
        "//bazel:linalg_lu_pivoting": ["LINALG_LU_PIVOTING=1"],
        "//conditions:default": [],
    })

    return defines

def linear_algebra_linkopts():
    return xsigma_linkopts()
