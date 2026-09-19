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
  target_compile_options(LinearAlgebra PRIVATE /WX)
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

if(LINALG_ENABLE_MKL)
  target_compile_definitions(LinearAlgebra PUBLIC LINALG_ENABLE_MKL)
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
