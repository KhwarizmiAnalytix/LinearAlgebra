load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for LinearAlgebra — mirrors CMake LINALG_CXX_STANDARD (default: 17)
LINALG_CXX_STD = "c++17"

def linear_algebra_copts():
    return xsigma_copts(cxx_std = LINALG_CXX_STD)

def linear_algebra_defines():
    """Returns compile definitions for linear_algebra.

    Mirrors CMakeLists.txt: LINALG_ENABLE_MKL / LINALG_LU_PIVOTING toggles.
    """
    defines = xsigma_defines()

    defines += select({
        "//bazel:linalg_enable_mkl": ["LINALG_ENABLE_MKL=1"],
        "//conditions:default": [],
    })
    defines += select({
        "//bazel:linalg_lu_pivoting": ["LINALG_LU_PIVOTING=1"],
        "//conditions:default": [],
    })

    return defines

def linear_algebra_linkopts():
    return xsigma_linkopts()
