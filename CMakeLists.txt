# =============================================================================
# LinearAlgebra —
# CMakeLists.txt
#
# Dense matrix decompositions (Cholesky, LU, QR, SVD, symmetric/general
# eigenvalue), linear solve, matrix inversion/multiplication/transpose/norm/
# rank/trace, least squares, pseudo-inverse, matrix exponential, and
# tri-/pentadiagonal operations, with optional Intel MKL, vendor-neutral
# CBLAS/LAPACKE, and cuBLAS/cuSOLVER GPU backends.
#
# Public compile definitions: LINALG_HAS_MKL — MKL-backed LAPACKE code
# paths; LINALG_HAS_BLAS — vendor-neutral CBLAS/LAPACKE code paths;
# LINALG_HAS_CUBLAS — cuBLAS/cuSOLVER GPU code paths under
# src/matrix_operation/gpu/cublas/. These (and LINALG_HAS_LAPACKE,
# LINALG_HAS_ACCELERATE, LINALG_HAS_VECTORIZATION, LINALG_HAS_LU_PIVOTING,
# LINALG_HAS_WIN32_WARNINGS below) are always-defined 0/1 macros, checked in
# C++ with plain #if (never #ifdef/#if defined), not conditionally-defined
# ones. Active for every CPU op except exponential and the
# tri-/pentadiagonal operations (scalar only), and every GPU op except
# general (non-symmetric) eigenvalue decomposition.
# =============================================================================
cmake_minimum_required(VERSION 3.16)

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
    include(CTest)
    enable_testing()
    option(BUILD_SHARED_LIBS "Build shared libraries" ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
else()
    set(LINALG_STANDALONE OFF)
endif()

# Own CMake modules: toolchain helpers (lto/cache/sanitize/...), the
# add_third_party_library machinery (third_party_helpers.cmake), and test
# dependency resolvers (LinearAlgebraDependencies.cmake) all live in Cmake/.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake")

# add_third_party_library + alias/folder helpers. When embedded in XSigma these
# are already defined by XSigma's ThirdParty/CMakeLists.txt; including the
# repo's own copy is a harmless identical redefinition.
include(third_party_helpers)

# --- Section 1: Feature flags (LINALG_ENABLE_*) ---

# Optional Intel MKL backend for the LAPACKE-based code paths in
# cholesky_decomposition / linear_solver / lu_decomposition /
# inversion / multiplication / transpose /
# svd_decomposition. Off by default: the portable scalar fallback
# implementations are used instead.
option(LINALG_ENABLE_MKL "Enable Intel MKL-backed LAPACKE code paths for LinearAlgebra" OFF)

# Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...).
# When both this and LINALG_ENABLE_MKL are ON, MKL takes precedence at
# compile time (see the #if/#elif chain in
# native/multiplication_native.cpp).
option(LINALG_ENABLE_BLAS "Enable a vendor-neutral CBLAS/LAPACKE backend for LinearAlgebra" OFF)

# cuBLAS (GEMM) / cuSOLVER (factorizations) GPU backend, exposed as
# linalg_backend::cuda_blas specializations of the same struct-template API
# used by the CPU backends (src/matrix_operation/gpu/cublas/). Requires the CUDA
# toolkit; enables the CUDA language only when this option is ON so plain CPU
# builds remain free of any CUDA toolchain requirement.
option(LINALG_ENABLE_CUBLAS "Enable a cuBLAS/cuSOLVER GPU backend for LinearAlgebra" OFF)

# Enables partial pivoting in the scalar (non-MKL) LU decomposition / solve /
# invert fallback paths. Off by default, preserving the existing fallback
# behavior exactly (matrix_operation's non-pivoting path is what currently
# ships).
option(LINALG_LU_PIVOTING "Enable partial pivoting in the scalar LU fallback paths" OFF)

# Enables the scalar (non-MKL/BLAS) multiplication backend's blocked,
# SIMD-vectorized GEMM (include/common/packet.h and the
# LINALG_HAS_VECTORIZATION block in native/multiplication_native.cpp) in
# place of its plain triple-loop fallback.
option(LINALG_ENABLE_VECTORIZATION
       "Enable the SIMD-vectorized scalar GEMM fallback for LinearAlgebra" ON
)

# Which linalg::packet<T> specialization backs LINALG_ENABLE_VECTORIZATION.
# There is no "auto" choice: the ISA is always explicit, and CMake computes a
# platform-appropriate default (below) when the cache variable is not
# overridden on the command line:
#   scalar - force linalg::packet<T>'s scalar (width-1) primary template.
#            Default for platforms other than x86_64/aarch64.
#   sse2/avx/avx2/avx512/neon - force that specialization and (x86 targets
#            only) add the matching -m<isa> compiler flag so it is actually
#            enabled; a target that cannot assemble the requested ISA fails
#            to build rather than silently falling back to a narrower one.
#            avx2 is the default on x86_64/AMD64/i386/i686; neon is the
#            default on aarch64/arm64.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _linalg_system_processor_lower)
if(_linalg_system_processor_lower MATCHES "^(aarch64|arm64)$")
    set(_linalg_vectorization_isa_default "neon")
elseif(_linalg_system_processor_lower MATCHES "^(x86_64|amd64|i386|i686)$")
    set(_linalg_vectorization_isa_default "avx2")
else()
    set(_linalg_vectorization_isa_default "scalar")
endif()
unset(_linalg_system_processor_lower)
set(LINALG_VECTORIZATION_ISA
    "${_linalg_vectorization_isa_default}"
    CACHE
        STRING
        "SIMD ISA for LinearAlgebra's vectorized scalar GEMM: scalar, sse2, avx, avx2, avx512, neon"
)
unset(_linalg_vectorization_isa_default)
set_property(
    CACHE LINALG_VECTORIZATION_ISA
    PROPERTY STRINGS
             scalar
             sse2
             avx
             avx2
             avx512
             neon
)

# On MSVC, native/multiplication_native.cpp normally suppresses C4244
# narrowing-conversion warnings around its non-MKL fallback; this re-enables
# them.
option(LINALG_DISPLAY_WIN32_WARNINGS "Do not suppress MSVC narrowing-conversion warnings" OFF)

# Link-Time Optimization — disabled when LINALG_ENABLE_COVERAGE or LINALG_ENABLE_SANITIZER is ON
# (LTO conflicts with instrumentation / sanitizers).
include(lto)
xsigma_lto_compute_default(_lto_default PROFILE SUPPORT)
set(LINALG_LTO_MODE "${_lto_default}"
    CACHE STRING "LTO mode for LinearAlgebra: off | thin | full | ipo | auto"
)
set_property(CACHE LINALG_LTO_MODE PROPERTY STRINGS off thin full ipo auto)

# Coverage instrumentation — activates -fprofile-instr-generate / --coverage compiler flags for
# this module only. LTO is automatically skipped when coverage is enabled.
option(LINALG_ENABLE_COVERAGE "Enable coverage instrumentation for LinearAlgebra" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
option(LINALG_ENABLE_TESTING "Build LinearAlgebra test suite" ON)

# Google Test integration for LinearAlgebra tests.
option(LINALG_ENABLE_GTEST "Enable GoogleTest for LinearAlgebra" ON)

# Google Benchmark integration for LinearAlgebra micro-benchmarks (vendored
# submodule under ThirdParty/benchmark; see linalg_setup_benchmark() in
# Cmake/LinearAlgebraDependencies.cmake). Off by default: no benchmark
# targets exist yet.
option(LINALG_ENABLE_BENCHMARK "Enable Google Benchmark for LinearAlgebra micro-benchmarks" OFF)

# Profiler-instrumented profiling program for all matrix operations (CPU and GPU).
# Requires Profiler::Profiler (auto-detected above via find_package(Profiler CONFIG QUIET)).
option(LINALG_ENABLE_PROFILING "Build the Profiler-instrumented matrix operation profiling program"
       OFF
)

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
include(CheckSymbolExists)

# b) Shared toolchain helpers
include(compiler_checks) # Detects compiler capabilities; sets PROJECT_* flags
include(checks) # Additional feature-check macros
include(build_type) # Validates / normalises CMAKE_BUILD_TYPE
include(platform) # Platform flags (macOS rpath, Windows /utf-8, …)

include(cache) # Configures per-target compiler cache launcher
include(CMakeDependentOption)
include(helper_macros)
if(LINALG_ENABLE_SANITIZER)
    include(sanitize)
endif()
if(LINALG_ENABLE_SPELL)
    include(spell)
endif()
if(LINALG_ENABLE_VALGRIND)
    include(valgrind)
endif()
# c) Coverage gate — appends -fprofile-instr-generate / --coverage; LTO already cleared above.
if(LINALG_ENABLE_COVERAGE)
    include(coverage)
elseif(MSVC)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /DEBUG:FULL")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " /DEBUG:FULL")
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
    find_package(MKL CONFIG QUIET)
    if(NOT MKL_FOUND)
        # Fall back to the classic (non-CONFIG) FindMKL-style search some
        # environments still ship instead of the oneMKL CMake config package.
        find_package(MKL QUIET)
    endif()
    if(NOT MKL_FOUND)
        message(
            FATAL_ERROR
                "LINALG_ENABLE_MKL is ON but MKL could not be found by find_package(MKL). "
                "Install Intel oneMKL or set MKLROOT / CMAKE_PREFIX_PATH, or configure with -DLINALG_ENABLE_MKL=OFF."
        )
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
        if(NOT MSVC AND NOT APPLE)
            list(APPEND LINALG_DEPENDENCY_LIBS -Wl,--no-as-needed)
        endif()
        list(APPEND LINALG_DEPENDENCY_LIBS MKL::MKL)
        if(NOT MSVC AND NOT APPLE)
            list(APPEND LINALG_DEPENDENCY_LIBS -Wl,--as-needed)
        endif()
    endif()
endif()

# --- Optional vendor-neutral BLAS/LAPACK backend ---
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
    # pick OpenBLAS up on their own, with no manual -DCMAKE_PREFIX_PATH
    # needed on macOS, Linux, or Windows. A user-supplied CMAKE_PREFIX_PATH
    # entry always wins ties, since list order is find preference and these
    # are appended, not prepended.
    if(NOT DEFINED ENV{LINALG_NO_AUTO_BLAS_PREFIX})
        set(LINALG_BLAS_PREFIX_CANDIDATES "")

        # Homebrew (macOS) / Linuxbrew (Linux) OpenBLAS keg: both are
        # keg-only, i.e. deliberately unlinked from the default prefix so
        # they don't shadow a system BLAS.
        find_program(LINALG_BREW_EXECUTABLE brew)
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
        mark_as_advanced(LINALG_BREW_EXECUTABLE)

        # Conda-forge's openblas package (macOS/Linux/Windows): headers and
        # library sit under the active environment's own prefix — Library/
        # on Windows, the prefix root elsewhere — never the compiler default.
        if(DEFINED ENV{CONDA_PREFIX})
            if(WIN32)
                list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}/Library")
            else()
                list(APPEND LINALG_BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}")
            endif()
        endif()

        # vcpkg's openblas port, for a VCPKG_ROOT set without the vcpkg
        # toolchain file (which would otherwise have already wired
        # CMAKE_PREFIX_PATH itself before this file ran).
        if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
            list(APPEND LINALG_BLAS_PREFIX_CANDIDATES
                 "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}"
            )
        endif()

        foreach(LINALG_BLAS_PREFIX_CANDIDATE IN LISTS LINALG_BLAS_PREFIX_CANDIDATES)
            if(LINALG_BLAS_PREFIX_CANDIDATE AND IS_DIRECTORY "${LINALG_BLAS_PREFIX_CANDIDATE}")
                list(APPEND CMAKE_PREFIX_PATH "${LINALG_BLAS_PREFIX_CANDIDATE}")
                message(
                    STATUS
                        "LinearAlgebra: found candidate OpenBLAS prefix ${LINALG_BLAS_PREFIX_CANDIDATE}"
                )
            endif()
        endforeach()
        unset(LINALG_BLAS_PREFIX_CANDIDATES)
        unset(LINALG_BLAS_PREFIX_CANDIDATE)
    endif()

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    # Imported BLAS::BLAS/LAPACK::LAPACK targets require CMake >= 3.18; fall
    # back to the raw *_LIBRARIES lists on older CMake (project minimum: 3.16).
    if(TARGET BLAS::BLAS)
        list(APPEND LINALG_DEPENDENCY_LIBS BLAS::BLAS)
    else()
        list(APPEND LINALG_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
    endif()
    if(TARGET LAPACK::LAPACK)
        list(APPEND LINALG_DEPENDENCY_LIBS LAPACK::LAPACK)
    else()
        list(APPEND LINALG_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
    endif()

    # BLAS::BLAS/LAPACK::LAPACK link the libraries but do not guarantee a
    # cblas.h/lapacke.h C declaration header is on the include path — Homebrew's
    # OpenBLAS, for instance, is keg-only and needs its prefix's include/ added
    # explicitly, which find_path() does via CMAKE_PREFIX_PATH/CMAKE_FIND_ROOT_PATH
    # (unlike CheckIncludeFileCXX, which only sees the compiler's default search
    # path). CMAKE_FIND_FRAMEWORK is forced to LAST for this lookup so a plain
    # header from the vendor find_package(BLAS) actually linked against (e.g.
    # OpenBLAS) is found ahead of an unrelated macOS framework's cblas.h —
    # mixing one vendor's declarations with a different vendor's linked
    # implementation is not guaranteed ABI-safe (e.g. differing CBLAS_INDEX /
    # ILP64 widths) and has produced silently wrong GEMM results in testing.
    set(_linalg_find_framework_save ${CMAKE_FIND_FRAMEWORK})
    set(CMAKE_FIND_FRAMEWORK LAST)
    find_path(LINALG_CBLAS_INCLUDE_DIR NAMES cblas.h)
    find_path(LINALG_LAPACKE_INCLUDE_DIR NAMES lapacke.h)
    set(CMAKE_FIND_FRAMEWORK ${_linalg_find_framework_save})
    unset(_linalg_find_framework_save)
    mark_as_advanced(LINALG_CBLAS_INCLUDE_DIR LINALG_LAPACKE_INCLUDE_DIR)

    set(LINALG_BLAS_INCLUDE_DIRS "")
    if(LINALG_CBLAS_INCLUDE_DIR MATCHES "\\.framework")
        # On macOS, the generic cblas.h search above commonly lands on
        # Accelerate's own vecLib.framework/Accelerate.framework header (Xcode's
        # SDK always exposes it) rather than a separate vendor's cblas.h. Route
        # through <Accelerate/Accelerate.h> in that case — found via the
        # compiler's default framework search path, no extra include dir needed
        # — so the ACCELERATE-specific code path (and LINALG_HAS_ACCELERATE) is
        # actually taken instead of silently falling through to the generic
        # <cblas.h> branch.
        set(LINALG_BLAS_USE_ACCELERATE ON)
    elseif(LINALG_CBLAS_INCLUDE_DIR)
        list(APPEND LINALG_BLAS_INCLUDE_DIRS "${LINALG_CBLAS_INCLUDE_DIR}")
    elseif(APPLE)
        include(CheckIncludeFileCXX)
        check_include_file_cxx("Accelerate/Accelerate.h" LINALG_HAVE_ACCELERATE_H)
        if(LINALG_HAVE_ACCELERATE_H)
            set(LINALG_BLAS_USE_ACCELERATE ON)
        endif()
    endif()
    if(NOT LINALG_CBLAS_INCLUDE_DIR AND NOT LINALG_BLAS_USE_ACCELERATE)
        message(
            FATAL_ERROR
                "LINALG_ENABLE_BLAS is ON but cblas.h (or, on Apple, Accelerate/Accelerate.h) "
                "was not found. Install a CBLAS-providing package (OpenBLAS, netlib, ...) "
                "and/or extend CMAKE_PREFIX_PATH, or configure with -DLINALG_ENABLE_BLAS=OFF."
        )
    endif()

    # Apple's Accelerate framework ships CBLAS (used for multiplication /
    # transpose) but deliberately does not ship the LAPACKE row-major C
    # wrapper — only the classic Fortran-mangled LAPACK entry points. The
    # LAPACKE-dependent blas_lapack backends (cholesky/lu/svd/inversion/
    # linear_solver) are therefore only compiled when real lapacke.h is found;
    # otherwise each op's dispatcher .cpp falls through to the scalar backend
    # at compile time via #if LINALG_HAS_LAPACKE.
    if(LINALG_LAPACKE_INCLUDE_DIR)
        # Debian/Ubuntu package the C interface separately from LAPACK.
        # Other vendors (including Homebrew OpenBLAS) bundle it in LAPACK.
        find_library(LINALG_LAPACKE_LIBRARY NAMES lapacke)
        mark_as_advanced(LINALG_LAPACKE_LIBRARY)
        if(LINALG_LAPACKE_LIBRARY)
            list(APPEND LINALG_DEPENDENCY_LIBS "${LINALG_LAPACKE_LIBRARY}")
        endif()
        list(APPEND LINALG_BLAS_INCLUDE_DIRS "${LINALG_LAPACKE_INCLUDE_DIR}")
        set(LINALG_BLAS_HAS_LAPACKE ON)
    else()
        message(STATUS "LinearAlgebra: lapacke.h not found — blas_lapack backend covers "
                       "multiplication/transpose only; cholesky/lu/svd/inversion/"
                       "linear_solver fall back to scalar under LINALG_ENABLE_BLAS."
        )
    endif()
    list(REMOVE_DUPLICATES LINALG_BLAS_INCLUDE_DIRS)
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
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD ${LINALG_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    list(APPEND LINALG_DEPENDENCY_LIBS CUDA::cudart CUDA::cublas CUDA::cusolver)
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

# --- Optional Profiler backend (KhwarizmiAnalytix/Profiler), auto-detected ---
# CPU/GPU profiling instrumentation via find_package(Profiler) — no vendored
# copy, no enable option: silently absent when not installed. Not yet wired
# into any call site; LINALG_HAS_PROFILER makes Profiler::Profiler available
# for a future instrumentation follow-up when the package happens to be
# present (e.g. built and installed from
# https://github.com/KhwarizmiAnalytix/Profiler).
find_package(Profiler CONFIG QUIET)
set(LINALG_HAS_PROFILER OFF)
if(TARGET Profiler::Profiler)
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
set(LOGGING_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(LOGGING_ENABLE_BENCHMARK OFF CACHE BOOL "" FORCE)
set(LOGGING_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/Logging"
    "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Logging_build" EXCLUDE_FROM_ALL
)
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

file(GLOB_RECURSE linalg_headers LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE linalg_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Native CPU implementation files under matrix_operation/cpu/native/ are included
# by the recursive glob above.
# The mkl/ and blas/ backend sources under src/matrix_operation/cpu/ include
# mkl.h / cblas.h / lapacke.h, so they are excluded from the recursive glob
# above and only appended — and only compiled — when LINALG_ENABLE_MKL /
# LINALG_ENABLE_BLAS turned on the matching backend above.
list(FILTER linalg_sources EXCLUDE REGEX "/src/matrix_operation/(cpu/(mkl|blas)|gpu/cublas)/")
if(LINALG_ENABLE_MKL)
    file(GLOB linalg_mkl_sources LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/cpu/mkl/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_mkl_sources})
endif()
if(LINALG_ENABLE_BLAS)
    file(GLOB linalg_blas_sources LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/cpu/blas/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_blas_sources})
endif()

# cuBLAS/cuSOLVER bridge sources and .cu kernel sources (all under
# src/matrix_operation/gpu/cublas/) need an nvcc toolchain, so they are only
# globbed in — and only compilable — when LINALG_ENABLE_CUBLAS turned on
# CUDA above.
if(LINALG_ENABLE_CUBLAS)
    file(GLOB linalg_cublas_sources LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_operation/gpu/cublas/*.cpp"
    )
    list(APPEND linalg_sources ${linalg_cublas_sources})
    file(GLOB_RECURSE linalg_cuda_sources LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
    list(APPEND linalg_sources ${linalg_cuda_sources})
endif()

# --- Target definition ---
set(CMAKE_CXX_STANDARD ${LINALG_CXX_STANDARD})

add_library(LinearAlgebra ${linalg_sources} ${linalg_headers})
xsigma_target_apply_lto(LinearAlgebra LINALG)
if(LINALG_ENABLE_SANITIZER)
    xsigma_configure_sanitizer(LINALG)
endif()
xsigma_find_linker(${LINALG_LINKER_CHOICE} LinearAlgebra LINALG_LTO_MODE)
xsigma_target_apply_cache(LinearAlgebra LINALG_ENABLE_CACHE LINALG_CACHE_BACKEND)
xsigma_target_optional_icecc(LinearAlgebra LINALG_ENABLE_ICECC)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(
        LinearAlgebra PUBLIC LINALG_SHARED_DEFINE PRIVATE LINALG_BUILDING_DLL
    )
    if(COMMAND xsigma_set_target_output_directories)
        xsigma_set_target_output_directories(LinearAlgebra)
    endif()
else()
    target_compile_definitions(LinearAlgebra PUBLIC LINALG_STATIC_DEFINE)
endif()

add_library(LinearAlgebra::LinearAlgebra ALIAS LinearAlgebra)
if(COMMAND xsigma_module_create_filters)
    xsigma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set_target_properties(
    LinearAlgebra PROPERTIES CXX_STANDARD ${LINALG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
                             CXX_EXTENSIONS OFF
)
if(MSVC)
    # Excludes CUDA: nvcc's frontend does not recognize the bare MSVC-style
    # /WX flag and misparses it as a second input file ("nvcc fatal: A single
    # input file is required for a non-link phase when an outputfile is
    # specified") when LINALG_ENABLE_CUBLAS compiles the .cu source below.
    target_compile_options(LinearAlgebra PRIVATE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/WX>)
else()
    target_compile_options(LinearAlgebra PRIVATE -include cstdlib)
endif()

# Include roots: repo root → "include/matrix_operation/lu_decomposition.h",
#                             "include/util/cpu_info.h", ...
target_include_directories(
    LinearAlgebra
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include/LinearAlgebra>
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
           "LINALG_HAS_CPUINFO=$<BOOL:${LINALG_HAS_CPUINFO}>"
           "LINALG_HAS_PROFILER=$<BOOL:${LINALG_HAS_PROFILER}>"
           "LINALG_HAS_LU_PIVOTING=$<BOOL:${LINALG_LU_PIVOTING}>"
           "LINALG_HAS_VECTORIZATION=$<BOOL:${LINALG_ENABLE_VECTORIZATION}>"
           "LINALG_HAS_WIN32_WARNINGS=$<BOOL:${LINALG_DISPLAY_WIN32_WARNINGS}>"
)
if(LINALG_ENABLE_VECTORIZATION)
    string(TOUPPER "${LINALG_VECTORIZATION_ISA}" _linalg_simd_isa_upper)
    target_compile_definitions(LinearAlgebra PUBLIC "LINALG_SIMD_${_linalg_simd_isa_upper}")
    unset(_linalg_simd_isa_upper)

    # x86 ISAs need the matching -m<isa> flag to actually be assembled;
    # NEON is the aarch64 baseline (and has no single standard x86-style
    # flag on the compilers this repository targets), so "neon" adds none.
    # MSVC has no per-ISA -mXXX flags of its own — /arch:AVX[2|512] is the
    # closest equivalent but covers the whole TU like GCC/Clang's -m flags
    # do, and this repository's MSVC support does not target AVX-512 at
    # all, so it is left for whoever first needs it to wire up.
    if(NOT MSVC)
        if(LINALG_VECTORIZATION_ISA STREQUAL "sse2")
            target_compile_options(LinearAlgebra PUBLIC -msse2)
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx")
            target_compile_options(LinearAlgebra PUBLIC -mavx)
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx2")
            target_compile_options(LinearAlgebra PUBLIC -mavx2 -mfma)
        elseif(LINALG_VECTORIZATION_ISA STREQUAL "avx512")
            # Half-packet operations also use the 128/256-bit FMA intrinsics.
            target_compile_options(LinearAlgebra PUBLIC -mavx512f -mfma)
        endif()
    endif()
endif()
if(LINALG_ENABLE_CLANGTIDY)
    include(clang_tidy)
    xsigma_target_clang_tidy(LinearAlgebra ${LINALG_ENABLE_FIX})
endif()
if(LINALG_ENABLE_IWYU)
    include(iwyu)
    xsigma_apply_iwyu(LinearAlgebra)
endif()
if(LINALG_DEPENDENCY_LIBS)
    target_link_libraries(LinearAlgebra PUBLIC ${LINALG_DEPENDENCY_LIBS})
endif()

# --- Installation ---
install(
    TARGETS LinearAlgebra
    EXPORT LinearAlgebraTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DESTINATION include/LinearAlgebra
    FILES_MATCHING
    PATTERN "*.h"
)

# --- CMake Package Configuration Files (find_package(LinearAlgebra) support) ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/LinearAlgebraConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LinearAlgebraConfig.cmake"
    INSTALL_DESTINATION lib/cmake/LinearAlgebra
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
        DESTINATION lib/cmake/LinearAlgebra
)

install(
    EXPORT LinearAlgebraTargets
    NAMESPACE LinearAlgebra::
    FILE LinearAlgebraTargets.cmake
    DESTINATION lib/cmake/LinearAlgebra
)

# --- Testing ---
if(LINALG_ENABLE_TESTING)
    add_subdirectory(Testing/Cxx)
endif()
if(LINALG_ENABLE_BENCHMARK)
    add_subdirectory(Testing/Benchmark)
endif()
if(LINALG_ENABLE_PROFILING)
    if(LINALG_HAS_PROFILER)
        add_subdirectory(Testing/Profile)
    else()
        message(WARNING "LINALG_ENABLE_PROFILING is ON but Profiler::Profiler was not found. "
                        "Build and install KhwarizmiAnalytix/Profiler and set CMAKE_PREFIX_PATH "
                        "to its install prefix, then re-configure."
        )
    endif()
endif()
if(LINALG_ENABLE_VALGRIND AND LINALG_ENABLE_TESTING)
    xsigma_apply_valgrind_timeouts()
endif()
if(LINALG_STANDALONE)
    message("")
    message("******** LinearAlgebra module (CMake flags) ********")
    message("  Shared libs          : ${BUILD_SHARED_LIBS}")
    message("  Cxx standard         : ${LINALG_CXX_STANDARD}")
    message("  Cache                : ${LINALG_ENABLE_CACHE}")
    message("  Cache backend        : ${LINALG_CACHE_BACKEND}")
    message("  Icecc                : ${LINALG_ENABLE_ICECC}")
    xsigma_print_lto_linker_summary(LINALG)
    message("  Coverage             : ${LINALG_ENABLE_COVERAGE}")
    message("  Testing              : ${LINALG_ENABLE_TESTING}")
    message("  Gtest                : ${LINALG_ENABLE_GTEST}")
    message("  Benchmark            : ${LINALG_ENABLE_BENCHMARK}")
    message("  Profiling            : ${LINALG_ENABLE_PROFILING}")
    message("  MKL                  : ${LINALG_ENABLE_MKL}")
    message("  BLAS/LAPACK          : ${LINALG_ENABLE_BLAS}")
    message("  Accelerate (BLAS)    : ${LINALG_BLAS_USE_ACCELERATE}")
    message("  LAPACKE (BLAS)       : ${LINALG_BLAS_HAS_LAPACKE}")
    message("  cuBLAS               : ${LINALG_ENABLE_CUBLAS}")
    message("  Cpuinfo              : ${LINALG_HAS_CPUINFO}")
    message("  Profiler             : ${LINALG_HAS_PROFILER}")
    message("  LU pivoting          : ${LINALG_LU_PIVOTING}")
    message("  Win32 warnings       : ${LINALG_DISPLAY_WIN32_WARNINGS}")
    message("  Vectorization        : ${LINALG_ENABLE_VECTORIZATION}")
    message("  Vectorization ISA    : ${LINALG_VECTORIZATION_ISA}")
    message("  Clang-tidy           : ${LINALG_ENABLE_CLANGTIDY}")
    message("  Fix                  : ${LINALG_ENABLE_FIX}")
    message("  Iwyu                 : ${LINALG_ENABLE_IWYU}")
    message("  Sanitizer            : ${LINALG_ENABLE_SANITIZER}")
    message("  Sanitizer type       : ${LINALG_SANITIZER_TYPE}")
    message("  Spell                : ${LINALG_ENABLE_SPELL}")
    message("  Valgrind             : ${LINALG_ENABLE_VALGRIND}")
    message("")
    message("Build:")
    message("  Default C++ standard : ${CMAKE_CXX_STANDARD}")
    message("  C flags              : ${CMAKE_C_FLAGS}")
    message("  CXX flags            : ${CMAKE_CXX_FLAGS}")
    message("  Shared linker flags  : ${CMAKE_SHARED_LINKER_FLAGS}")
    message("  Static linker flags  : ${CMAKE_STATIC_LINKER_FLAGS}")
    message("  Module linker flags  : ${CMAKE_MODULE_LINKER_FLAGS}")
    get_directory_property(
        _summary_compile_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS
    )
    message("  Top-level compile defs : ${_summary_compile_defs}")
    message("")
endif()
