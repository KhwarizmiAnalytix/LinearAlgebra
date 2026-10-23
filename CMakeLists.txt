# =============================================================================
# LinearAlgebra —
# CMakeLists.txt
#
# Dense matrix decompositions (Cholesky, LU, QR, SVD, symmetric/general
# eigenvalue), linear solve, matrix inversion/multiplication/transpose/norm/
# rank/trace, least squares, pseudo-inverse, matrix exponential, and
# tri-/pentadiagonal operations, with optional Intel MKL, vendor-neutral
# CBLAS/LAPACKE, cuBLAS/cuSOLVER, hipBLAS/hipSOLVER, and Metal Performance
# Shaders GPU backends.
#
# Public compile definitions: LINALG_HAS_MKL — MKL-backed LAPACKE code
# paths; LINALG_HAS_BLAS — vendor-neutral CBLAS/LAPACKE code paths;
# LINALG_HAS_CUBLAS — cuBLAS/cuSOLVER GPU code paths under
# src/matrix_operation/gpu/cublas/; LINALG_HAS_ROCBLAS — hipBLAS/hipSOLVER
# GPU code paths under src/matrix_operation/gpu/hipblas/; LINALG_HAS_METAL —
# Metal Performance Shaders GPU code paths under
# src/matrix_operation/gpu/metal/ (float only; multiplication, batched
# multiplication, transpose, norm, and trace only — see linalg_backend::
# METAL_BLAS's doc comment in include/common/backend.h). These (and
# LINALG_HAS_LAPACKE, LINALG_HAS_ACCELERATE, LINALG_HAS_VECTORIZATION,
# LINALG_HAS_LU_PIVOTING, LINALG_HAS_WIN32_WARNINGS below) are
# always-defined 0/1 macros, checked in C++ with plain #if (never
# #ifdef/#if defined), not conditionally-defined ones. Active for every CPU
# op except exponential and the tri-/pentadiagonal operations (scalar
# only), and every GPU op except general (non-symmetric) eigenvalue
# decomposition (METAL_BLAS covers a narrower subset still, see above).
# LINALG_ENABLE_CUBLAS, LINALG_ENABLE_ROCBLAS, and LINALG_ENABLE_METAL are
# pairwise mutually exclusive (see the guard below): all three target a
# single `batched_matrix_multiplication` free-function symbol
# (include/matrix_operation/multiplication_batched.h), which is not
# backend-templated.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# Standalone CMake package — any C++ project can consume it via add_subdirectory;
# XSigma is one consumer, not a required host.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        LinearAlgebra
        VERSION 1.0.0
        DESCRIPTION
            "Dense linear algebra: Cholesky/LU/SVD decompositions, linear solve, matrix inversion/multiplication/transpose, optional MKL backend"
        HOMEPAGE_URL "https://github.com/KhwarizmiAnalytix/LinearAlgebra"
        LANGUAGES CXX C
    )
    set(LINALG_STANDALONE ON)
    option(BUILD_SHARED_LIBS "Build shared libraries" ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
else()
    set(LINALG_STANDALONE OFF)
endif()

set(LINALG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include(GNUInstallDirs)

# Own CMake modules: toolchain helpers (lto/cache/sanitize/...), the
# linalg_add_third_party_library machinery (third_party_helpers.cmake), and test
# dependency resolvers (LinearAlgebraDependencies.cmake) all live in Cmake/.

# Project-owned dependency alias and folder helpers.
include("${LINALG_SOURCE_DIR}/Cmake/third_party_helpers.cmake")

# --- Section 1: Feature flags (LINALG_ENABLE_*) ---

# Optional Intel MKL backend for the LAPACKE-based code paths in
# cholesky_decomposition / linear_solver / lu_decomposition /
# inversion / multiplication / transpose /
# svd_decomposition. Off by default: the portable scalar fallback
# implementations are used instead.
option(LINALG_ENABLE_MKL "Enable Intel MKL-backed LAPACKE code paths for LinearAlgebra" OFF)

# Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...).
# When MKL is also enabled, load this provider through a private runtime handle
# so identical CBLAS/LAPACKE names cannot bind to MKL's ILP64 interface.
option(LINALG_ENABLE_BLAS "Enable a vendor-neutral CBLAS/LAPACKE backend for LinearAlgebra" OFF)

# Explicit BLIS backend option (FLAME/AOCL-compatible CBLAS/LAPACKE provider).
# When ON, prefix probing targets BLIS before falling through to find_package(BLAS).
# LINALG_ENABLE_BLAS is activated automatically; enable only one of BLIS or BLAS.
# BLIS does not ship LAPACKE — pair with AOCL libflame or netlib LAPACKE for
# factorization ops (cholesky / lu / svd / inversion / linear_solver).
option(LINALG_ENABLE_BLIS "Enable BLIS as the CBLAS/LAPACKE provider" OFF)

# cuBLAS (GEMM) / cuSOLVER (factorizations) GPU backend, exposed as
# linalg_backend::CUDA_BLAS specializations of the same struct-template API
# used by the CPU backends (src/matrix_operation/gpu/cublas/). Requires the CUDA
# toolkit; enables the CUDA language only when this option is ON so plain CPU
# builds remain free of any CUDA toolchain requirement.
option(LINALG_ENABLE_CUBLAS "Enable a cuBLAS/cuSOLVER GPU backend for LinearAlgebra" OFF)

# hipBLAS (GEMM) / hipSOLVER (factorizations) GPU backend, exposed as
# linalg_backend::ROCM_BLAS specializations of the same struct-template API
# used by the CPU and CUDA_BLAS backends (src/matrix_operation/gpu/hipblas/).
# Requires a ROCm/HIP toolchain; enables the HIP language only when this
# option is ON so plain CPU/CUDA builds remain free of any ROCm toolchain
# requirement. Mutually exclusive with LINALG_ENABLE_CUBLAS (see the guard
# below multiplication_batched's free functions are not backend-templated).
option(LINALG_ENABLE_ROCBLAS "Enable a hipBLAS/hipSOLVER GPU backend for LinearAlgebra" OFF)

# Metal Performance Shaders GPU backend, exposed as linalg_backend::METAL_BLAS
# specializations of the same struct-template API used by the CPU and other
# GPU backends (src/matrix_operation/gpu/metal/). Apple platforms only
# (enforced below); float only, since Apple GPUs have no native double
# type, and covering only multiplication / batched multiplication /
# transpose / norm / trace, since Metal Performance Shaders has no QR/SVD/
# general-solver routines (see linalg_backend::METAL_BLAS's doc comment in
# include/common/backend.h and docs/backend_roadmap.md §3, which deferred
# Apple MPS support as "float-only ... high effort" before this backend
# existed). Mutually exclusive with LINALG_ENABLE_CUBLAS/LINALG_ENABLE_ROCBLAS
# (see the guard below).
option(
    LINALG_ENABLE_METAL
    "Enable a Metal Performance Shaders GPU backend (float-only; Apple platforms) for LinearAlgebra"
    OFF
)

if(LINALG_ENABLE_METAL AND NOT APPLE)
    message(
        FATAL_ERROR
            "LINALG_ENABLE_METAL requires an Apple platform (Metal is not available elsewhere)."
    )
endif()

if((LINALG_ENABLE_CUBLAS AND LINALG_ENABLE_ROCBLAS) OR (LINALG_ENABLE_CUBLAS AND LINALG_ENABLE_METAL
                                                       ) OR (LINALG_ENABLE_ROCBLAS
                                                             AND LINALG_ENABLE_METAL)
)
    message(
        FATAL_ERROR
            "LINALG_ENABLE_CUBLAS, LINALG_ENABLE_ROCBLAS, and LINALG_ENABLE_METAL are "
            "pairwise mutually exclusive: every GPU backend defines the same "
            "batched_matrix_multiplication symbols "
            "(include/matrix_operation/multiplication_batched.h is not "
            "backend-templated), so enabling more than one would be a duplicate-symbol "
            "link error. Enable only the GPU backend that matches the target hardware."
    )
endif()

option(LINALG_BUILD_EXAMPLES "Build the CPU/GPU backend benchmark example" OFF)

# Enables partial pivoting in the scalar (non-MKL) LU decomposition / solve /
# invert fallback paths. Off by default, preserving the existing fallback
# behavior exactly (matrix_operation's non-pivoting path is what currently
# ships).
option(LINALG_LU_PIVOTING "Enable partial pivoting in the scalar LU fallback paths" OFF)

# Enables the scalar (non-MKL/BLAS) multiplication backend's blocked,
# SIMD-vectorized GEMM (src/common/packet.h and the
# LINALG_HAS_VECTORIZATION block in native/multiplication_native.cpp) in
# place of its plain triple-loop fallback.
option(LINALG_ENABLE_VECTORIZATION
       "Enable the SIMD-vectorized scalar GEMM fallback for LinearAlgebra" ON
)

# On MSVC, native/multiplication_native.cpp normally suppresses C4244
# narrowing-conversion warnings around its non-MKL fallback; this re-enables
# them.
option(LINALG_DISPLAY_WIN32_WARNINGS "Do not suppress MSVC narrowing-conversion warnings" OFF)

# Link-Time Optimization — disabled when LINALG_ENABLE_COVERAGE or LINALG_ENABLE_SANITIZER is ON
# (LTO conflicts with instrumentation / sanitizers).
include("${LINALG_SOURCE_DIR}/Cmake/lto.cmake")
linalg_lto_compute_default(_lto_default PROFILE SUPPORT)
set(LINALG_LTO_MODE "${_lto_default}"
    CACHE STRING "LTO mode for LinearAlgebra: off | thin | full | ipo | auto"
)
set_property(CACHE LINALG_LTO_MODE PROPERTY STRINGS off thin full ipo auto)

# Coverage instrumentation — activates -fprofile-instr-generate / --coverage compiler flags for
# this module only. LTO is automatically skipped when coverage is enabled.
option(LINALG_ENABLE_COVERAGE "Enable coverage instrumentation for LinearAlgebra" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
set(_linalg_testing_default ${LINALG_STANDALONE})
if(DEFINED BUILD_TESTING)
    set(_linalg_testing_default ${BUILD_TESTING})
endif()
option(LINALG_ENABLE_TESTING "Build LinearAlgebra test suite" ${_linalg_testing_default})
# BUILD_TESTING=OFF is the host's global veto, including on reconfigure.
if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    set(LINALG_ENABLE_TESTING OFF)
endif()

# Google Test integration for LinearAlgebra tests.
option(LINALG_ENABLE_GTEST "Enable GoogleTest for LinearAlgebra" ON)

# Google Benchmark integration for LinearAlgebra micro-benchmarks (vendored
# submodule under ThirdParty/benchmark; see linalg_setup_benchmark() in
# Cmake/LinearAlgebraDependencies.cmake). Off by default.
option(LINALG_ENABLE_BENCHMARK "Enable Google Benchmark for LinearAlgebra micro-benchmarks" OFF)

# Profiler instrumentation and profiling programs for all matrix operations (CPU and GPU).
# Requires Profiler::Profiler when enabled.
option(LINALG_ENABLE_PROFILING "Enable Profiler instrumentation and build profiling programs" OFF)

# Icecream distributed compilation launcher (rarely needed; hidden by default).
option(LINALG_ENABLE_ICECC "Use Icecream (icecc) distributed compiler for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_ICECC)

# Per-target compiler cache — wraps every compile command with ccache / sccache etc. to speed up
# incremental and CI rebuilds. ON by default.
option(LINALG_ENABLE_CACHE "Enable compiler caching for the LinearAlgebra target" ON)
mark_as_advanced(LINALG_ENABLE_CACHE)

option(LINALG_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_CLANGTIDY)
option(LINALG_ENABLE_FIX "Enable clang-tidy fix-errors and fix options for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_FIX)
option(LINALG_ENABLE_IWYU "Enable include-what-you-use analysis for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_IWYU)
option(LINALG_ENABLE_SANITIZER "Build LinearAlgebra with sanitizer support (Clang/GCC only)" OFF)
set(LINALG_SANITIZER_TYPE "address"
    CACHE STRING
          "Sanitizer type for LinearAlgebra. Options: address, undefined, thread, memory, leak"
)
set_property(CACHE LINALG_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)
mark_as_advanced(LINALG_ENABLE_SANITIZER LINALG_SANITIZER_TYPE)
option(LINALG_ENABLE_SPELL "Enable check-only spell checking for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_SPELL)
option(LINALG_ENABLE_VALGRIND "Execute LinearAlgebra test suite with Valgrind" OFF)
mark_as_advanced(LINALG_ENABLE_VALGRIND)
set(LINALG_LINKER_CHOICE "default"
    CACHE STRING "Linker for LinearAlgebra: default, lld, mold, gold, lld-link"
)
set_property(CACHE LINALG_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)
mark_as_advanced(LINALG_LINKER_CHOICE)
set(LINALG_CACHE_BACKEND "none"
    CACHE STRING
          "Compiler cache backend for LinearAlgebra. Options: none, ccache, sccache, buildcache"
)
set_property(CACHE LINALG_CACHE_BACKEND PROPERTY STRINGS none ccache sccache buildcache)
mark_as_advanced(LINALG_CACHE_BACKEND)

# C++ standard — applied to every TU in this module via CMAKE_CXX_STANDARD.
set(LINALG_CXX_STANDARD "17" CACHE STRING "C++ standard for LinearAlgebra. Options: 17 20 23")
set_property(CACHE LINALG_CXX_STANDARD PROPERTY STRINGS 17 20 23)
if(NOT LINALG_CXX_STANDARD MATCHES "^(17|20|23)$")
    message(FATAL_ERROR "LINALG_CXX_STANDARD must be one of: 17 20 23")
endif()

# --- Section 2: CMake module includes --- Order: introspection → toolchain helpers →
# coverage/sanitizer → LTO/linker.

# a) Standard CMake introspection
include(CheckCXXSourceRuns)
include(CheckCXXSymbolExists)

# b) Shared toolchain helpers
# Compiler minimum versions and system feature checks.
include("${LINALG_SOURCE_DIR}/Cmake/checks.cmake")
include("${LINALG_SOURCE_DIR}/Cmake/build_type.cmake") # Validates / normalises CMAKE_BUILD_TYPE
# Platform compilation flags.
include("${LINALG_SOURCE_DIR}/Cmake/platform.cmake")

include("${LINALG_SOURCE_DIR}/Cmake/cache.cmake") # Configures per-target compiler cache launcher
include(CMakeDependentOption)
include("${LINALG_SOURCE_DIR}/Cmake/helper_macros.cmake")
if(LINALG_ENABLE_SANITIZER)
    include("${LINALG_SOURCE_DIR}/Cmake/sanitize.cmake")
endif()
if(LINALG_ENABLE_SPELL)
    include("${LINALG_SOURCE_DIR}/Cmake/spell.cmake")
endif()
if(LINALG_ENABLE_VALGRIND)
    include("${LINALG_SOURCE_DIR}/Cmake/valgrind.cmake")
endif()
if(LINALG_STANDALONE)
    include(CTest)
endif()
if(LINALG_ENABLE_TESTING)
    enable_testing()
endif()
if(LINALG_ENABLE_COVERAGE)
    include("${LINALG_SOURCE_DIR}/Cmake/coverage.cmake")
endif()

# --- Optional Intel MKL backend ---
set(LINALG_DEPENDENCY_LIBS "")
if(LINALG_ENABLE_MKL)
    # MKL call sites pass linalg_int/linalg_long (64-bit) straight through as
    # MKL_INT* with no narrowing, so the oneMKL CMake config package must
    # select its ilp64 (64-bit MKL_INT) interface, not the lp64 default.
    # Must be set before find_package(MKL CONFIG) — it selects which
    # MKL::MKL interface target that package defines.
    set(MKL_INTERFACE "ilp64")
    find_package(MKL CONFIG REQUIRED)
    if(NOT TARGET MKL::MKL)
        message(FATAL_ERROR "Intel oneMKL must provide MKL::MKL with the ILP64 interface.")
    endif()
    if(TARGET MKL::MKL)
        # GNU ld's default --as-needed can drop libmkl_core.so (and the
        # threading library) from DT_NEEDED: the interface library
        # (libmkl_intel_ilp64.so) calls into them through an internal ABI
        # rather than a linker-visible undefined symbol, so --as-needed sees no
        # reason to keep the dependency and the loader later fails with
        # "undefined symbol: mkl_blas_dgemm" the first time MKL is actually
        # called. Intel's own linking guidance is to force --no-as-needed
        # around the MKL libraries on ELF/GNU-ld platforms; MSVC has no such
        # flag and Apple's ld doesn't prune unresolved lazy symbols this way.
        if(UNIX AND NOT APPLE)
            list(APPEND LINALG_DEPENDENCY_LIBS -Wl,--no-as-needed)
        endif()
        list(APPEND LINALG_DEPENDENCY_LIBS MKL::MKL)
        if(UNIX AND NOT APPLE)
            list(APPEND LINALG_DEPENDENCY_LIBS -Wl,--as-needed)
        endif()
    endif()
endif()

# --- Optional vendor-neutral BLAS/LAPACK backend ---
# LINALG_ENABLE_BLIS activates the same blas_lapacke backend as
# LINALG_ENABLE_BLAS; it just changes which provider is probed for.
if(LINALG_ENABLE_BLIS AND NOT LINALG_ENABLE_BLAS)
    set(LINALG_ENABLE_BLAS ON)
endif()
set(LINALG_BLAS_USE_ACCELERATE OFF)
set(LINALG_BLAS_HAS_LAPACKE OFF)
if(LINALG_ENABLE_BLAS)
    # Package managers that install OpenBLAS outside the compiler's default
    # search path leave find_package(BLAS) to silently resolve to whatever
    # the platform ships instead (e.g. Accelerate on Apple, which has no
    # lapacke.h), so the blas_lapack backend's LAPACKE-dependent ops
    # (cholesky/lu/svd/inversion/linear_solver) fall back to scalar. Probe
    # each manager's own prefix query / env var — PyTorch's
    # cmake/Modules/FindOpenBLAS.cmake takes the same approach — and append
    # any hit to CMAKE_PREFIX_PATH (not CMAKE_FIND_ROOT_PATH, which would
    # also constrain *what* can be found) so find_package/find_path below
    # pick the right vendor up on their own, with no manual -DCMAKE_PREFIX_PATH
    # needed on macOS, Linux, or Windows. A user-supplied CMAKE_PREFIX_PATH
    # entry always wins ties, since list order is find preference and these
    # are appended, not prepended.
    set(LINALG_BLAS_INCLUDE_DIRS "")
    set(_linalg_blis_found_via_config OFF)
    if(LINALG_ENABLE_BLIS)
        set(BLA_VENDOR FLAME)
        find_package(BLIS CONFIG QUIET)
        if(BLIS_FOUND AND (TARGET BLIS::BLIS OR BLIS_LIBRARIES))
            if(TARGET BLIS::BLIS)
                set(_linalg_blas_libraries BLIS::BLIS)
            else()
                set(_linalg_blas_libraries ${BLIS_LIBRARIES})
            endif()
            list(APPEND LINALG_BLAS_INCLUDE_DIRS ${BLIS_INCLUDE_DIRS})
            set(_linalg_blis_found_via_config ON)
        endif()
    endif()
    if(NOT CMAKE_CROSSCOMPILING AND NOT DEFINED ENV{LINALG_NO_AUTO_BLAS_PREFIX})
        set(LINALG_BLAS_PREFIX_CANDIDATES "")
        find_program(LINALG_BREW_EXECUTABLE brew)
        mark_as_advanced(LINALG_BREW_EXECUTABLE)

        if(LINALG_ENABLE_BLIS)
            if(NOT _linalg_blis_found_via_config)
                # Homebrew / Linuxbrew blis keg (keg-only on macOS).
                if(LINALG_BREW_EXECUTABLE)
                    execute_process(
                        COMMAND "${LINALG_BREW_EXECUTABLE}" --prefix blis
                        OUTPUT_VARIABLE _linalg_brew_blis_prefix
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                        RESULT_VARIABLE _linalg_brew_blis_result
                    )
                    if(_linalg_brew_blis_result EQUAL 0)
                        list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "${_linalg_brew_blis_prefix}")
                    endif()
                endif()

                # Conda-forge blis package.
                if(DEFINED ENV{CONDA_PREFIX})
                    if(WIN32)
                        list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}/Library")
                    else()
                        list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}")
                    endif()
                endif()

                # vcpkg blis port.
                if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
                    list(APPEND LINALG_BLAS_PREFIX_CANDIDATES
                         "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}"
                    )
                endif()
            endif()
        else()
            # OpenBLAS (default CBLAS provider).
            # Homebrew (macOS) / Linuxbrew (Linux) OpenBLAS keg: both are
            # keg-only, i.e. deliberately unlinked from the default prefix so
            # they don't shadow a system BLAS.
            if(LINALG_BREW_EXECUTABLE)
                execute_process(
                    COMMAND "${LINALG_BREW_EXECUTABLE}" --prefix openblas
                    OUTPUT_VARIABLE LINALG_HOMEBREW_OPENBLAS_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                    RESULT_VARIABLE LINALG_HOMEBREW_OPENBLAS_RESULT
                )
                if(LINALG_HOMEBREW_OPENBLAS_RESULT EQUAL 0)
                    list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "${LINALG_HOMEBREW_OPENBLAS_PREFIX}")
                endif()
            endif()

            # Conda-forge's openblas package (macOS/Linux/Windows).
            if(DEFINED ENV{CONDA_PREFIX})
                if(WIN32)
                    list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}/Library")
                else()
                    list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}")
                endif()
            endif()

            # vcpkg's openblas port.
            if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
                list(APPEND LINALG_BLAS_PREFIX_CANDIDATES
                     "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}"
                )
            endif()
        endif()

        foreach(LINALG_BLAS_PREFIX_CANDIDATE IN LISTS LINALG_BLAS_PREFIX_CANDIDATES)
            if(LINALG_BLAS_PREFIX_CANDIDATE AND IS_DIRECTORY "${LINALG_BLAS_PREFIX_CANDIDATE}")
                list(APPEND CMAKE_PREFIX_PATH "${LINALG_BLAS_PREFIX_CANDIDATE}")
                message(
                    STATUS
                        "LinearAlgebra: found candidate BLAS prefix ${LINALG_BLAS_PREFIX_CANDIDATE}"
                )
            endif()
        endforeach()
        unset(LINALG_BLAS_PREFIX_CANDIDATES)
        unset(LINALG_BLAS_PREFIX_CANDIDATE)
    endif()

    # The generic backend uses 32-bit CBLAS integers. MKL has a separate ILP64 interface.
    if(DEFINED BLA_SIZEOF_INTEGER AND NOT BLA_SIZEOF_INTEGER STREQUAL "4"
       AND NOT BLA_SIZEOF_INTEGER STREQUAL "ANY"
    )
        message(FATAL_ERROR "The vendor-neutral BLAS backend requires BLA_SIZEOF_INTEGER=4")
    endif()
    set(BLA_SIZEOF_INTEGER 4)
    if(LINALG_ENABLE_MKL AND (NOT DEFINED BLA_VENDOR OR BLA_VENDOR STREQUAL "All"))
        set(BLA_VENDOR OpenBLAS)
    endif()
    if(NOT _linalg_blis_found_via_config)
        find_package(BLAS REQUIRED)
        set(_linalg_blas_libraries BLAS::BLAS)
    endif()
    if(LINALG_ENABLE_BLIS)
        # BLIS supplies CBLAS; its optional LAPACK companion can be libFLAME or
        # netlib. Avoid FindLAPACK rediscovering a second BLAS implementation.
        find_library(LINALG_BLIS_LAPACK_LIBRARY NAMES flame lapack)
        if(LINALG_BLIS_LAPACK_LIBRARY)
            list(APPEND _linalg_blas_libraries "${LINALG_BLIS_LAPACK_LIBRARY}")
        endif()
    else()
        find_package(LAPACK REQUIRED)
        list(APPEND _linalg_blas_libraries LAPACK::LAPACK)
    endif()
    include("${LINALG_SOURCE_DIR}/Cmake/blas.cmake")
    linalg_check_blas_provider()
    if(LINALG_ENABLE_MKL)
        include("${LINALG_SOURCE_DIR}/Cmake/isolated_blas.cmake")
        linalg_configure_isolated_blas()
        list(APPEND LINALG_DEPENDENCY_LIBS ${CMAKE_DL_LIBS})
    else()
        list(APPEND LINALG_DEPENDENCY_LIBS ${_linalg_blas_libraries})
    endif()
endif()

# --- Optional cuBLAS / cuSOLVER GPU backend ---
# cuBLAS/cuSOLVER themselves are host-callable C APIs, linked into ordinary
# .cpp translation units with no CUDA toolchain needed.
# Several operations under src/matrix_operation/gpu/cublas/ are genuine
# CUDA-kernel .cu files (batched matmul, norm, QR, SVD, pseudo-inverse,
# inversion/determinant, trace), so `enable_language(CUDA)` and an nvcc
# toolchain are only required — and only enabled — when this option is ON;
# plain CPU builds remain free of any CUDA toolchain requirement.
if(LINALG_ENABLE_CUBLAS)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD ${LINALG_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                            STREQUAL "GNU"
    )
        # Mixed CXX/CUDA targets link with clang++. CMake's CUDA runtime
        # options are bare cudadevrt/cudart names, which the MSVC linker
        # frontend turns into *.lib. The GNU driver treats those as input
        # files instead of searching the CUDA library directory.
        set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED
            "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib" "${CUDAToolkit_LIBRARY_DIR}/cudart.lib"
        )
        set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC
            "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
            "${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib"
        )
    endif()
    list(APPEND LINALG_DEPENDENCY_LIBS CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# --- Optional hipBLAS / hipSOLVER GPU backend ---
# hipBLAS/hipSOLVER themselves are host-callable C APIs, linked into ordinary
# .cpp translation units with no HIP toolchain needed.
# Several operations under src/matrix_operation/gpu/hipblas/ are genuine
# HIP-kernel .hip files (batched matmul, norm, QR, SVD, pseudo-inverse,
# inversion/determinant, trace — the same set as their cuBLAS .cu
# counterparts), so `enable_language(HIP)` and a ROCm toolchain are only
# required — and only enabled — when this option is ON; plain CPU/CUDA
# builds remain free of any ROCm toolchain requirement.
if(LINALG_ENABLE_ROCBLAS)
    find_package(hip CONFIG REQUIRED)
    find_package(hipblas CONFIG REQUIRED)
    find_package(hipsolver CONFIG REQUIRED)
    if(NOT DEFINED CMAKE_HIP_STANDARD)
        set(CMAKE_HIP_STANDARD ${LINALG_CXX_STANDARD})
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
    endif()
    enable_language(HIP)
    # CMake's HIP language supplies device compilation/linking for .hip files.
    # hip::device instead injects -x hip/--hip-link into ordinary C++ sources
    # and consumers, which may use GCC rather than ROCm's Clang.
    list(APPEND LINALG_DEPENDENCY_LIBS hip::host roc::hipblas roc::hipsolver)
endif()

# --- Optional Metal Performance Shaders GPU backend ---
# Metal/MetalPerformanceShaders are Objective-C frameworks, so the bridge
# sources under src/matrix_operation/gpu/metal/ are Objective-C++ (.mm); a
# plain C++ (.cpp) source lives alongside them for trace, which needs no
# Metal API calls at all (see trace_metal.cpp). OBJCXX is CMake's own
# language for .mm files (Clang-only, which every Apple toolchain ships),
# enabled only when this option is ON so plain CPU/CUDA/HIP builds need no
# Objective-C++ support at all.
if(LINALG_ENABLE_METAL)
    enable_language(OBJCXX)
    find_library(LINALG_METAL_FRAMEWORK Metal REQUIRED)
    find_library(LINALG_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(LINALG_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    list(APPEND LINALG_DEPENDENCY_LIBS ${LINALG_METAL_FRAMEWORK} ${LINALG_MPS_FRAMEWORK}
         ${LINALG_FOUNDATION_FRAMEWORK}
    )
endif()

# --- Optional cpuinfo backend (pytorch/cpuinfo), auto-detected ---
# CPU cache-size introspection via find_package(cpuinfo) — no vendored copy,
# no enable option: silently absent when not installed. When found,
# LINALG_HAS_CPUINFO gates include/util/cpu_info.h's cpuinfo_cach() to read
# real L1/L2/L3 sizes from pytorch/cpuinfo instead of its hardcoded
# conservative defaults (used when the package is not installed).
find_package(cpuinfo CONFIG QUIET)
set(LINALG_HAS_CPUINFO OFF)
if(TARGET cpuinfo::cpuinfo)
    list(APPEND LINALG_DEPENDENCY_LIBS cpuinfo::cpuinfo)
    set(LINALG_HAS_CPUINFO ON)
elseif(TARGET cpuinfo)
    list(APPEND LINALG_DEPENDENCY_LIBS cpuinfo)
    set(LINALG_HAS_CPUINFO ON)
endif()

# --- Optional Profiler backend (KhwarizmiAnalytix/Profiler) ---
# LINALG_ENABLE_PROFILING controls discovery, linkage, and scope instrumentation.
set(LINALG_HAS_PROFILER OFF)
if(LINALG_ENABLE_PROFILING)
    if(NOT TARGET Profiler::Profiler)
        find_package(Profiler CONFIG REQUIRED)
    endif()
    list(APPEND LINALG_DEPENDENCY_LIBS Profiler::Profiler)
    set(LINALG_HAS_PROFILER ON)
endif()

# --- Logging (required) ---
# KhwarizmiAnalytix/Logging (ThirdParty/Logging, a nested submodule with its
# own fmt/loguru/glog/spdlog/magic_enum submodules) provides logging::
# exception, the type every LOGGING_THROW call site in this library throws
# (include/matrix_operation/) — there is no
# std::runtime_error fallback, so this is unconditional, not gated behind an
# option. Logging is a full standalone CMake package (same
# add_subdirectory-embeddable shape as this repo itself); disable its own
# test/benchmark subtrees when embedded here — LinearAlgebra has its own
# GoogleTest wiring and no use for Logging's benchmark_logging_logger target.
option(LINALG_USE_SYSTEM_LOGGING "Resolve Logging from an installed config package" OFF)
include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")
set(_linalg_owns_logging OFF)
if(NOT TARGET Logging::Logging)
    if(LINALG_USE_SYSTEM_LOGGING)
        find_package(Logging CONFIG REQUIRED)
    else()
        # Scope these choices to our bundled dependency configuration.
        function(linalg_add_logging)
            # Bundled Logging performs C probes and builds C dependencies. Enable
            # that language here so an otherwise C++-only host works unchanged.
            if(NOT CMAKE_C_COMPILER_LOADED)
                enable_language(C)
            endif()
            set(LOGGING_ENABLE_TESTING OFF)
            set(LOGGING_ENABLE_BENCHMARK OFF)
            set(LOGGING_ENABLE_EXAMPLES OFF)
            set(XSIGMA_ENABLE_EXTERNAL "${LINALG_ENABLE_EXTERNAL}")
            # Loguru treats an empty CMAKE_DL_LIBS as missing stack-trace
            # support. macOS supplies dladdr in libSystem and a libdl stub;
            # use that stub within this dependency's scope to keep stack
            # traces enabled without changing the bundled sources.
            if(APPLE AND NOT CMAKE_DL_LIBS)
                set(CMAKE_DL_LIBS dl)
            endif()
            linalg_add_dependency_subdirectory(
                "${LINALG_SOURCE_DIR}/ThirdParty/Logging"
                "${LINALG_BINARY_DIR}/ThirdParty/Logging_build" EXCLUDE_FROM_ALL
            )
        endfunction()
        linalg_add_logging()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND MSVC_VERSION LESS 1920)
            include("${LINALG_SOURCE_DIR}/Cmake/LoggingMsvcCompatibility.cmake")
            linalg_apply_logging_msvc_compatibility()
        endif()
        set(_linalg_owns_logging ON)
    endif()
endif()
if(_linalg_owns_logging
   AND WIN32
   AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
)
    get_target_property(_linalg_logging_libraries Logging LINK_LIBRARIES)
    list(TRANSFORM _linalg_logging_libraries REPLACE "^dbghelp$" "-ldbghelp")
    set_property(TARGET Logging PROPERTY LINK_LIBRARIES "${_linalg_logging_libraries}")
endif()
# Keep the in-tree target out of this export set, and link the separately
# installed Logging package for consumers (including static library users).
list(APPEND LINALG_DEPENDENCY_LIBS $<BUILD_INTERFACE:Logging::Logging>
     $<INSTALL_INTERFACE:Logging::Logging>
)

# --- Configure-time headers ---
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Testing/LinearAlgebraTest.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Testing/LinearAlgebraTest.h" @ONLY
)

# --- Source discovery ---
set(linalg_headers "")
set(linalg_sources "")

file(GLOB_RECURSE linalg_headers CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)
file(GLOB_RECURSE linalg_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Native CPU implementation files under matrix_operation/cpu/native/ are included
# by the recursive glob above.
# The mkl/ and blas/ backend sources under src/matrix_operation/cpu/ include
# mkl.h / cblas.h / lapacke.h, so they are excluded from the recursive glob
# above and only appended — and only compiled — when LINALG_ENABLE_MKL /
# LINALG_ENABLE_BLAS turned on the matching backend above.
list(FILTER linalg_sources EXCLUDE REGEX
     "/src/matrix_operation/(cpu/(mkl|blas)|gpu/(cublas|hipblas|metal))/"
)
if(LINALG_ENABLE_MKL)
    file(GLOB linalg_mkl_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/cpu/mkl/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_mkl_sources})
endif()
if(LINALG_ENABLE_BLAS)
    file(GLOB linalg_blas_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/cpu/blas/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_blas_sources})
endif()

# cuBLAS/cuSOLVER bridge sources and .cu kernel sources (all under
# src/matrix_operation/gpu/cublas/) need an nvcc toolchain, so they are only
# globbed in — and only compilable — when LINALG_ENABLE_CUBLAS turned on
# CUDA above.
if(LINALG_ENABLE_CUBLAS)
    file(GLOB linalg_cublas_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/gpu/cublas/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_cublas_sources})
    file(GLOB_RECURSE linalg_cuda_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
    list(APPEND linalg_sources ${linalg_cuda_sources})
endif()

# hipBLAS/hipSOLVER bridge sources and .hip kernel sources (all under
# src/matrix_operation/gpu/hipblas/) need a ROCm/HIP toolchain, so they are
# only globbed in — and only compilable — when LINALG_ENABLE_ROCBLAS turned
# on HIP above.
if(LINALG_ENABLE_ROCBLAS)
    file(GLOB linalg_hipblas_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/gpu/hipblas/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_hipblas_sources})
    file(GLOB_RECURSE linalg_hip_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hip"
    )
    list(APPEND linalg_sources ${linalg_hip_sources})
endif()

# Metal bridge sources (all under src/matrix_operation/gpu/metal/) need an
# Objective-C++ (.mm) compiler, so they are only globbed in -- and only
# compilable -- when LINALG_ENABLE_METAL turned on OBJCXX above. trace's
# Metal source is plain C++ (no Metal API calls, see trace_metal.cpp) but
# lives alongside its .mm siblings and is globbed in the same way.
#
# The .mm files use ordinary alloc/init Objective-C object lifetimes (no
# explicit `release`, relying on @autoreleasepool for the per-call
# MPSMatrix/MPSMatrixMultiplication objects and on leaking the lazily-
# created process-wide device/queue/pipeline-state singletons on purpose),
# so they require ARC -- `enable_language(OBJCXX)` alone does not turn it
# on -- scoped to just these sources so the rest of the library is
# unaffected.
if(LINALG_ENABLE_METAL)
    file(GLOB linalg_metal_mm_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/gpu/metal/*.mm"
    )
    # Linking a framework adds -F, which makes SDK headers look like project
    # headers. Classify the Metal SDK search path as a system framework path
    # so Apple's own deprecated declarations do not produce build warnings.
    get_filename_component(_linalg_metal_framework_dir "${LINALG_METAL_FRAMEWORK}" DIRECTORY)
    set_source_files_properties(
        ${linalg_metal_mm_sources}
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc;-iframework;${_linalg_metal_framework_dir}"
    )
    list(APPEND linalg_sources ${linalg_metal_mm_sources})
    file(GLOB linalg_metal_cpp_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/gpu/metal/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_metal_cpp_sources})
endif()

# --- Target definition ---
set(CMAKE_CXX_STANDARD ${LINALG_CXX_STANDARD})

add_library(LinearAlgebra ${linalg_sources} ${linalg_headers})
target_compile_features(LinearAlgebra PUBLIC cxx_std_${LINALG_CXX_STANDARD})
linalg_target_apply_cache(LinearAlgebra LINALG_ENABLE_CACHE LINALG_CACHE_BACKEND)
linalg_target_optional_icecc(LinearAlgebra LINALG_ENABLE_ICECC)
if(LINALG_ENABLE_SANITIZER)
    linalg_configure_sanitizer(LinearAlgebra LINALG)
endif()
if(LINALG_ENABLE_COVERAGE)
    linalg_configure_coverage(LinearAlgebra)
endif()
linalg_target_apply_lto(LinearAlgebra LINALG)
linalg_find_linker(${LINALG_LINKER_CHOICE} LinearAlgebra)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(
        LinearAlgebra PUBLIC LINALG_SHARED_DEFINE PRIVATE LINALG_BUILDING_DLL
    )
    if(COMMAND linalg_set_target_output_directories)
        linalg_set_target_output_directories(LinearAlgebra)
    endif()
else()
    target_compile_definitions(LinearAlgebra PUBLIC LINALG_STATIC_DEFINE)
endif()

add_library(LinearAlgebra::LinearAlgebra ALIAS LinearAlgebra)
if(linalg_sources)
    source_group(TREE "${LINALG_SOURCE_DIR}" FILES ${linalg_headers} ${linalg_sources})
endif()

set_target_properties(
    LinearAlgebra PROPERTIES CXX_STANDARD ${LINALG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
                             CXX_EXTENSIONS OFF
)
linalg_apply_platform_compile_flags(LinearAlgebra)

# Include roots: repo root → "include/matrix_operation/lu_decomposition.h",
#                             "include/util/cpu_info.h", ...
target_include_directories(
    LinearAlgebra
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/LinearAlgebra>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(LINALG_BLAS_INCLUDE_DIRS)
    target_include_directories(LinearAlgebra PRIVATE ${LINALG_BLAS_INCLUDE_DIRS})
endif()
# OpenBLAS lapack.h uses float _Complex / double _Complex (C99), which MSVC
# does not support in C++ mode.  The fix is to include <complex> before
# <lapacke.h> in each BLAS source file so std::complex<> is available when
# lapack_complex_float / lapack_complex_double are expanded.  Per-file
# compile properties scope the macros to the BLAS TUs only so MKL sources
# (which use mkl_lapacke.h with its own struct-based complex types) are not
# affected.
if(LINALG_ENABLE_BLAS AND MSVC)
    file(GLOB _linalg_blas_srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/cpu/blas/*.cpp")
    set_source_files_properties(
        ${_linalg_blas_srcs}
        PROPERTIES
            COMPILE_DEFINITIONS
            "lapack_complex_float=std::complex<float>;lapack_complex_double=std::complex<double>"
    )
    unset(_linalg_blas_srcs)
endif()
target_compile_definitions(
    LinearAlgebra
    PUBLIC "LINALG_HAS_MKL=$<BOOL:${LINALG_ENABLE_MKL}>"
           "LINALG_HAS_BLAS=$<BOOL:${LINALG_ENABLE_BLAS}>"
           "LINALG_HAS_LAPACKE=$<BOOL:${LINALG_BLAS_HAS_LAPACKE}>"
           "LINALG_HAS_ACCELERATE=$<BOOL:${LINALG_BLAS_USE_ACCELERATE}>"
           "LINALG_HAS_CUBLAS=$<BOOL:${LINALG_ENABLE_CUBLAS}>"
           "LINALG_HAS_ROCBLAS=$<BOOL:${LINALG_ENABLE_ROCBLAS}>"
           "LINALG_HAS_METAL=$<BOOL:${LINALG_ENABLE_METAL}>"
           "LINALG_HAS_CPUINFO=$<BOOL:${LINALG_HAS_CPUINFO}>"
           "LINALG_HAS_PROFILER=$<BOOL:${LINALG_HAS_PROFILER}>"
           "LINALG_HAS_LU_PIVOTING=$<BOOL:${LINALG_LU_PIVOTING}>"
           "LINALG_HAS_VECTORIZATION=$<BOOL:${LINALG_ENABLE_VECTORIZATION}>"
           "LINALG_HAS_WIN32_WARNINGS=$<BOOL:${LINALG_DISPLAY_WIN32_WARNINGS}>"
)
linalg_apply_vectorization_flags(LinearAlgebra)
if(LINALG_ENABLE_CLANGTIDY)
    include("${LINALG_SOURCE_DIR}/Cmake/clang_tidy.cmake")
    linalg_target_clang_tidy(LinearAlgebra ${LINALG_ENABLE_FIX})
endif()
if(LINALG_ENABLE_IWYU)
    include("${LINALG_SOURCE_DIR}/Cmake/iwyu.cmake")
    linalg_apply_iwyu(LinearAlgebra)
endif()
if(LINALG_DEPENDENCY_LIBS)
    target_link_libraries(LinearAlgebra PUBLIC ${LINALG_DEPENDENCY_LIBS})
endif()

# --- Installation ---
install(
    TARGETS LinearAlgebra
    EXPORT LinearAlgebraTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LinearAlgebra
    FILES_MATCHING
    PATTERN "*.h"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/MPL-2.0.txt"
              "${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/matrix_functions.NOTICE"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/LinearAlgebra/licenses
)

# --- CMake Package Configuration Files (find_package(LinearAlgebra) support) ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/LinearAlgebraConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LinearAlgebra
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfigVersion.cmake"
    # Hardcoded rather than ${PROJECT_VERSION}: when embedded via
    # add_subdirectory (LINALG_STANDALONE OFF), PROJECT_VERSION belongs to
    # whichever project() call is nearest/outermost, not this one's
    # project(LinearAlgebra VERSION 1.0.0 ...), which only runs in the
    # standalone branch above. Keep in sync with that VERSION.
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LinearAlgebra
)

install(
    EXPORT LinearAlgebraTargets
    NAMESPACE LinearAlgebra::
    FILE LinearAlgebraTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LinearAlgebra
)

# --- Testing ---
include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")
linalg_setup_eigen()
if(LINALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LINALG_ENABLE_TESTING)
    add_subdirectory(Testing/Cxx)
endif()
if(LINALG_ENABLE_BENCHMARK)
    add_subdirectory(Testing/Benchmark)
endif()
if(LINALG_ENABLE_PROFILING)
    add_subdirectory(Testing/Profile)
endif()
if(LINALG_ENABLE_VALGRIND AND LINALG_ENABLE_TESTING)
    linalg_apply_valgrind_timeouts()
endif()

# --- Configuration summary ---
include("${LINALG_SOURCE_DIR}/Cmake/summary.cmake")
