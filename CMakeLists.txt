# =============================================================================
# LinearAlgebra —
# CMakeLists.txt
#
# Dense matrix decompositions (Cholesky, LU, SVD), linear solve, and matrix
# inversion/multiplication/transpose, with an optional Intel MKL backend.
#
# Public compile definitions: LINALG_ENABLE_MKL — MKL-backed LAPACKE code
# paths active in cholesky/LU/SVD/inversion/multiplication/transpose.
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# Standalone CMake package — any C++ project can consume it via add_subdirectory;
# XSigma is one consumer, not a required host.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(
    LinearAlgebra
    VERSION 1.0.0
    DESCRIPTION "Dense linear algebra: Cholesky/LU/SVD decompositions, linear solve, matrix inversion/multiplication/transpose, optional MKL backend"
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
# matrix_inversion / matrix_multiplication / matrix_transpose /
# svd_decomposition. Off by default: the portable scalar fallback
# implementations are used instead.
option(LINALG_ENABLE_MKL "Enable Intel MKL-backed LAPACKE code paths for LinearAlgebra" OFF)

# Vendor-neutral CBLAS/LAPACKE backend (OpenBLAS, Accelerate, netlib, ...).
# When both this and LINALG_ENABLE_MKL are ON, MKL takes precedence at
# compile time (see e.g. matrix_multiplication.cxx's #if/#elif chain).
option(LINALG_ENABLE_BLAS "Enable a vendor-neutral CBLAS/LAPACKE backend for LinearAlgebra" OFF)

# cuBLAS (GEMM) / cuSOLVER (factorizations) GPU backend, exposed as separate
# linalg::gpu::* entry points (include/matrix_operation_gpu/) rather than a
# device parameter on the CPU ops. Requires the CUDA toolkit; enables the
# CUDA language only when this option is ON so plain CPU builds need no CUDA
# toolchain.
option(LINALG_ENABLE_CUBLAS "Enable a cuBLAS/cuSOLVER GPU backend for LinearAlgebra" OFF)

# Enables partial pivoting in the scalar (non-MKL) LU decomposition / solve /
# invert fallback paths. Off by default, preserving the existing fallback
# behavior exactly (matrix_operation's non-pivoting path is what currently
# ships).
option(LINALG_LU_PIVOTING "Enable partial pivoting in the scalar LU fallback paths" OFF)

# On MSVC, matrix_multiplication.cxx normally suppresses narrowing-conversion
# warnings (C4244) around its non-MKL fallback; this re-enables them.
option(LINALG_DISPLAY_WIN32_WARNINGS "Do not suppress MSVC narrowing-conversion warnings" OFF)

# Link-Time Optimization — disabled when LINALG_ENABLE_COVERAGE or LINALG_ENABLE_SANITIZER is ON
# (LTO conflicts with instrumentation / sanitizers).
include(lto)
xsigma_lto_compute_default(_lto_default PROFILE SUPPORT)
set(LINALG_LTO_MODE "${_lto_default}" CACHE STRING
                                            "LTO mode for LinearAlgebra: off | thin | full | ipo | auto"
)
set_property(CACHE LINALG_LTO_MODE PROPERTY STRINGS off thin full ipo auto)

# Coverage instrumentation — activates -fprofile-instr-generate / --coverage compiler flags for
# this module only. LTO is automatically skipped when coverage is enabled.
option(LINALG_ENABLE_COVERAGE "Enable coverage instrumentation for LinearAlgebra" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
option(LINALG_ENABLE_TESTING "Build LinearAlgebra test suite" ON)

# Google Test integration for LinearAlgebra tests.
option(LINALG_ENABLE_GTEST "Enable GoogleTest for LinearAlgebra" ON)

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
    CACHE STRING "Sanitizer type for LinearAlgebra. Options: address, undefined, thread, memory, leak"
)
set_property(CACHE LINALG_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)
mark_as_advanced(LINALG_ENABLE_SANITIZER LINALG_SANITIZER_TYPE)
option(LINALG_ENABLE_SPELL "Enable check-only spell checking for LinearAlgebra" OFF)
mark_as_advanced(LINALG_ENABLE_SPELL)
option(LINALG_ENABLE_VALGRIND "Execute LinearAlgebra test suite with Valgrind" OFF)
mark_as_advanced(LINALG_ENABLE_VALGRIND)
set(LINALG_LINKER_CHOICE "default" CACHE STRING
                                          "Linker for LinearAlgebra: default, lld, mold, gold, lld-link"
)
set_property(CACHE LINALG_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)
mark_as_advanced(LINALG_LINKER_CHOICE)
set(LINALG_CACHE_BACKEND "none"
    CACHE STRING "Compiler cache backend for LinearAlgebra. Options: none, ccache, sccache, buildcache"
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
    list(APPEND LINALG_DEPENDENCY_LIBS MKL::MKL)
  endif()
endif()

# --- Optional vendor-neutral BLAS/LAPACK backend ---
if(LINALG_ENABLE_BLAS)
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

  set(LINALG_BLAS_USE_ACCELERATE OFF)
  set(LINALG_BLAS_INCLUDE_DIRS "")
  if(LINALG_CBLAS_INCLUDE_DIR)
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

  # Apple's Accelerate framework ships CBLAS (used for matrix_multiplication /
  # matrix_transpose) but deliberately does not ship the LAPACKE row-major C
  # wrapper — only the classic Fortran-mangled LAPACK entry points. The
  # LAPACKE-dependent blas_lapack backends (cholesky/lu/svd/inversion/
  # linear_solver) are therefore only compiled when real lapacke.h is found;
  # otherwise each op's dispatcher .cxx falls through to the scalar backend
  # at compile time via #if LINALG_BLAS_HAS_LAPACKE.
  set(LINALG_BLAS_HAS_LAPACKE OFF)
  if(LINALG_LAPACKE_INCLUDE_DIR)
    list(APPEND LINALG_BLAS_INCLUDE_DIRS "${LINALG_LAPACKE_INCLUDE_DIR}")
    set(LINALG_BLAS_HAS_LAPACKE ON)
  else()
    message(
      STATUS
      "LinearAlgebra: lapacke.h not found — blas_lapack backend covers "
      "matrix_multiplication/matrix_transpose only; cholesky/lu/svd/inversion/"
      "linear_solver fall back to scalar under LINALG_ENABLE_BLAS."
    )
  endif()
  list(REMOVE_DUPLICATES LINALG_BLAS_INCLUDE_DIRS)
endif()

# --- Optional cuBLAS / cuSOLVER GPU backend ---
# cuBLAS/cuSOLVER themselves are host-callable C APIs, linked into ordinary
# .cxx translation units with no CUDA toolchain needed. One op
# (matrix_multiplication_batched_gpu.cu, batched square-matrix multiply) is a
# genuine hand-written CUDA kernel, so `enable_language(CUDA)` and an nvcc
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

# --- Configure-time headers ---
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/common/configure.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/common/configure.h" @ONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Testing/LinearAlgebraTest.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Testing/LinearAlgebraTest.h" @ONLY
)

# --- Source discovery ---
set(linalg_headers "")
set(linalg_sources "")

file(GLOB_RECURSE linalg_headers LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE linalg_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cxx")

# .cu sources (currently just matrix_multiplication_batched_gpu.cu) need an
# nvcc toolchain, so they are only globbed in — and only compilable — when
# LINALG_ENABLE_CUBLAS turned on CUDA above.
if(LINALG_ENABLE_CUBLAS)
  file(GLOB_RECURSE linalg_cuda_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cu")
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
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_SHARED_DEFINE PRIVATE LINALG_BUILDING_DLL)
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
  LinearAlgebra PROPERTIES CXX_STANDARD ${LINALG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS
                                                                                   OFF
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
#                             "include/util/exception.h", ...
# Build tree     → generated "include/common/configure.h"
target_include_directories(
  LinearAlgebra
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:include/LinearAlgebra>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(LINALG_BLAS_INCLUDE_DIRS)
  target_include_directories(LinearAlgebra PRIVATE ${LINALG_BLAS_INCLUDE_DIRS})
endif()

if(LINALG_ENABLE_MKL)
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_ENABLE_MKL)
endif()
if(LINALG_ENABLE_BLAS)
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_ENABLE_BLAS)
endif()
if(LINALG_ENABLE_CUBLAS)
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_ENABLE_CUBLAS)
endif()
if(LINALG_LU_PIVOTING)
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_LU_PIVOTING)
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
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/include/common/configure.h"
  DESTINATION include/LinearAlgebra/include/common
)

# --- Testing ---
if(LINALG_ENABLE_TESTING)
  add_subdirectory(Testing/Cxx)
endif()

if(LINALG_ENABLE_VALGRIND AND LINALG_ENABLE_TESTING)
  xsigma_apply_valgrind_timeouts()
endif()

message("")
message("******** LinearAlgebra module (CMake flags) ********")
message("  Cxx standard     : ${LINALG_CXX_STANDARD}")
message("  Cache            : ${LINALG_ENABLE_CACHE}")
message("  Cache backend    : ${LINALG_CACHE_BACKEND}")
message("  Icecc            : ${LINALG_ENABLE_ICECC}")
xsigma_print_lto_linker_summary(LINALG)
message("  Coverage         : ${LINALG_ENABLE_COVERAGE}")
message("  Testing          : ${LINALG_ENABLE_TESTING}")
message("  Gtest            : ${LINALG_ENABLE_GTEST}")
message("  MKL              : ${LINALG_ENABLE_MKL}")
message("  BLAS/LAPACK      : ${LINALG_ENABLE_BLAS}")
message("  cuBLAS/cuSOLVER  : ${LINALG_ENABLE_CUBLAS}")
message("  LU pivoting      : ${LINALG_LU_PIVOTING}")
message("  Clang-tidy       : ${LINALG_ENABLE_CLANGTIDY}")
message("  Fix              : ${LINALG_ENABLE_FIX}")
message("  Iwyu             : ${LINALG_ENABLE_IWYU}")
message("  Sanitizer        : ${LINALG_ENABLE_SANITIZER}")
message("  Sanitizer type   : ${LINALG_SANITIZER_TYPE}")
message("  Spell            : ${LINALG_ENABLE_SPELL}")
message("  Valgrind         : ${LINALG_ENABLE_VALGRIND}")
message("")
message("Build:")
message("  Default C++ standard     : ${CMAKE_CXX_STANDARD}")
message("  C flags                  : ${CMAKE_C_FLAGS}")
message("  CXX flags                : ${CMAKE_CXX_FLAGS}")
message("  Shared linker flags      : ${CMAKE_SHARED_LINKER_FLAGS}")
message("  Static linker flags      : ${CMAKE_STATIC_LINKER_FLAGS}")
message("  Module linker flags      : ${CMAKE_MODULE_LINKER_FLAGS}")
get_directory_property(_summary_compile_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
message("  Top-level compile defs   : ${_summary_compile_defs}")
message("")
