# =============================================================================
# XSigma System
# Validation and Checks Module
# =============================================================================
# This module performs efficient system validation with aggressive caching to minimize CMake
# reconfiguration overhead while ensuring all required dependencies and capabilities are available.
#
# Performance Optimizations: - Cached validation results to avoid redundant checks - Fast platform
# detection with cached results - Efficient compiler capability validation - Streamlined dependency
# checking

# Validate compiler minimums on every configure, including when feature checks are cached.
# The floor scales with the selected C++ standard: each later standard needs a strictly
# newer compiler to even recognize the -std=/-std:c++NN flag, let alone support the
# language — a compiler that only clears the C++17 floor fails later with a raw
# "unrecognized command line option" instead of this clear FATAL_ERROR.
if(DEFINED LINALG_CXX_STANDARD AND NOT LINALG_CXX_STANDARD STREQUAL "")
    set(_linalg_checks_min_standard "${LINALG_CXX_STANDARD}")
elseif(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL "")
    set(_linalg_checks_min_standard "${CMAKE_CXX_STANDARD}")
else()
    set(_linalg_checks_min_standard 20)
endif()

if(_linalg_checks_min_standard STREQUAL "23")
    # -std=c++23 (previously -std=c++2b).
    # https://gcc.gnu.org/gcc-11/changes.html
    set(_linalg_checks_min_gcc 11.0)
    # https://releases.llvm.org/17.0.0/tools/clang/docs/ReleaseNotes.html
    set(_linalg_checks_min_clang 17.0)
    # Apple has not published an explicit minimum for -std=c++23; best available estimate
    # — verify against CI before relying on it.
    set(_linalg_checks_min_appleclang 15.0)
    # /std:c++23preview from VS 17.0 (19.30); stable /std:c++23 only in very recent toolsets.
    # https://learn.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version
    set(_linalg_checks_min_msvc 19.34)
elseif(_linalg_checks_min_standard STREQUAL "20")
    # -std=c++20 (previously only -std=c++2a).
    # https://gcc.gnu.org/gcc-10/changes.html
    set(_linalg_checks_min_gcc 10.0)
    # https://releases.llvm.org/11.0.0/tools/clang/docs/ReleaseNotes.html
    set(_linalg_checks_min_clang 11.0)
    # Xcode 12 (AppleClang 12) still rejects -std=c++20 ("use c++2a"); Xcode 14
    # (AppleClang 14) accepts it — 13 is a conservative floor pending CI confirmation.
    set(_linalg_checks_min_appleclang 13.0)
    # https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/
    set(_linalg_checks_min_msvc 19.29)
else()
    # C++17 baseline. Floor is set by <filesystem> (used unconditionally by Logging's
    # string_util.h), not just language support: GCC 7.1/Clang 5.0 only have
    # <experimental/filesystem> and lack the <filesystem> header entirely ("fatal error:
    # filesystem: No such file or directory"), and GCC < 9.1 / Clang(LLVM) < 9.0 need an
    # extra -lstdc++fs/-lc++fs link flag this build does not add.
    # https://gcc.gnu.org/gcc-9/changes.html
    set(_linalg_checks_min_gcc 9.1)
    # https://releases.llvm.org/9.0.0/projects/libcxx/docs/UsingLibcxx.html
    set(_linalg_checks_min_clang 9.0)
    # AppleClang 11 / Xcode 11 is the first to ship the non-experimental <filesystem>
    # header; AppleClang 10 / Xcode 10 only has <experimental/filesystem>.
    # https://en.cppreference.com/w/cpp/compiler_support/17
    set(_linalg_checks_min_appleclang 11.0)
    # MSVC 19.14 / VS 2017 15.7 also supports /Zc:__cplusplus, used by platform.cmake.
    # https://learn.microsoft.com/en-us/cpp/overview/what-s-new-for-cpp-2017
    set(_linalg_checks_min_msvc 19.14)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS
                                            _linalg_checks_min_gcc
)
    message(
        FATAL_ERROR
            "GCC ${_linalg_checks_min_gcc} or later is required for C++${_linalg_checks_min_standard}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS
                                              _linalg_checks_min_clang
)
    message(
        FATAL_ERROR
            "LLVM Clang ${_linalg_checks_min_clang} or later is required for C++${_linalg_checks_min_standard}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS
                                                   _linalg_checks_min_appleclang
)
    message(
        FATAL_ERROR
            "Apple Clang ${_linalg_checks_min_appleclang} or later is required for C++${_linalg_checks_min_standard}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS
                                             _linalg_checks_min_msvc
)
    message(
        FATAL_ERROR
            "MSVC ${_linalg_checks_min_msvc} or later is required for C++${_linalg_checks_min_standard}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

unset(_linalg_checks_min_standard)
unset(_linalg_checks_min_gcc)
unset(_linalg_checks_min_clang)
unset(_linalg_checks_min_appleclang)
unset(_linalg_checks_min_msvc)

include_guard(DIRECTORY)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)
find_package(Threads REQUIRED)
if(WIN32)
    check_include_file_cxx("windows.h" LINALG_HAVE_WINDOWS_H)
    if(NOT LINALG_HAVE_WINDOWS_H)
        message(FATAL_ERROR "LinearAlgebra requires windows.h on Windows.")
    endif()
elseif(UNIX)
    check_include_file_cxx("unistd.h" LINALG_HAVE_UNISTD_H)
    check_include_file_cxx("pthread.h" LINALG_HAVE_PTHREAD_H)
    if(NOT LINALG_HAVE_UNISTD_H OR NOT LINALG_HAVE_PTHREAD_H)
        message(FATAL_ERROR "LinearAlgebra requires unistd.h and pthread.h.")
    endif()
endif()
