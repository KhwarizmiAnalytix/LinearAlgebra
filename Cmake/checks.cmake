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
# https://gcc.gnu.org/gcc-7/changes.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.1)
    message(
        FATAL_ERROR "GCC 7.1 or later is required for C++17. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

# https://clang.llvm.org/cxx_status.html#cxx17
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(
        FATAL_ERROR
            "LLVM Clang 5.0 or later is required for C++17. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

# AppleClang 10 / Xcode 10 adds the C++17 <optional>, <any>, and <variant> headers.
# https://developer.apple.com/documentation/xcode-release-notes/xcode-10-release-notes/
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    message(
        FATAL_ERROR
            "Apple Clang 10.0 (Xcode 10) or later is required for C++17. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

# MSVC 19.14 / VS 2017 15.7 also supports /Zc:__cplusplus, used by platform.cmake.
# https://learn.microsoft.com/en-us/cpp/overview/what-s-new-for-cpp-2017
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.14)
    message(
        FATAL_ERROR
            "MSVC 19.14 (Visual Studio 2017 15.7) or later is required for C++17. Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

# Guard against multiple inclusions for performance
if(TEMP_CHECKS_CONFIGURED)
  return()
endif()
set(TEMP_CHECKS_CONFIGURED TRUE CACHE INTERNAL "Checks module loaded")

# Create cache key for validation results
set(TEMP_VALIDATION_CACHE_KEY
    "${CMAKE_CXX_COMPILER_ID}_${CMAKE_CXX_COMPILER_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_VERSION}"
    CACHE INTERNAL "Validation cache key"
)

# Check if validation has already been completed for this configuration
if(TEMP_VALIDATION_COMPLETED STREQUAL TEMP_VALIDATION_CACHE_KEY)
  message(STATUS "XSigma: Using cached validation results")
  return()
endif()

message(STATUS "XSigma: Performing system validation...")

# =============================================================================
# Platform
# Detection with Caching

# Fast platform detection with cached results
if(NOT DEFINED TEMP_PLATFORM_DETECTED)
  if(WIN32)
    set(TEMP_PLATFORM "Windows" CACHE INTERNAL "Detected platform")
    set(TEMP_PLATFORM_WINDOWS TRUE CACHE INTERNAL "Windows platform detected")
  elseif(APPLE)
    set(TEMP_PLATFORM "macOS" CACHE INTERNAL "Detected platform")
    set(TEMP_PLATFORM_MACOS TRUE CACHE INTERNAL "macOS platform detected")
  elseif(UNIX)
    set(TEMP_PLATFORM "Linux" CACHE INTERNAL "Detected platform")
    set(TEMP_PLATFORM_LINUX TRUE CACHE INTERNAL "Linux platform detected")
  else()
    message(WARNING "XSigma: Unknown platform detected")
    set(TEMP_PLATFORM "Unknown" CACHE INTERNAL "Detected platform")
  endif()
  set(TEMP_PLATFORM_DETECTED TRUE CACHE INTERNAL "Platform detection completed")
  message(STATUS "XSigma: Platform detected: ${TEMP_PLATFORM}")
endif()

# =============================================================================
# Compiler Detection
# Version minimums are enforced above.

set(TEMP_COMPILER_ID ${CMAKE_CXX_COMPILER_ID} CACHE INTERNAL "Compiler ID")
mark_as_advanced(TEMP_COMPILER_ID)

# GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(TEMP_COMPILER_GCC TRUE CACHE INTERNAL "GCC compiler detected")
  message(STATUS "XSigma: GCC ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(TEMP_COMPILER_ID "gcc" CACHE INTERNAL "Compiler ID")

  # Clang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(TEMP_COMPILER_CLANG TRUE CACHE INTERNAL "Clang compiler detected")
  message(STATUS "XSigma: Clang ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(TEMP_COMPILER_ID "clang" CACHE INTERNAL "Compiler ID")

  # Apple Clang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(TEMP_COMPILER_APPLE_CLANG TRUE CACHE INTERNAL "Apple Clang compiler detected")
  message(STATUS "XSigma: Apple Clang ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(TEMP_COMPILER_ID "clang" CACHE INTERNAL "Compiler ID")

  # MSVC
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(TEMP_COMPILER_MSVC TRUE CACHE INTERNAL "MSVC compiler detected")
  message(STATUS "XSigma: MSVC ${CMAKE_CXX_COMPILER_VERSION} validated")
  set(TEMP_COMPILER_ID "msvc" CACHE INTERNAL "Compiler ID")

else()
  message(WARNING "XSigma: Unknown compiler '${CMAKE_CXX_COMPILER_ID}'. Build may fail.")
endif()

# =============================================================================
# Essential System
# Dependencies Validation

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckSymbolExists)

# Threading support validation (cached)
if(NOT DEFINED TEMP_THREADING_VALIDATED)
  find_package(Threads REQUIRED)
  if(NOT Threads_FOUND)
    message(FATAL_ERROR "XSigma: Threading support is required but not found")
  endif()
  set(TEMP_THREADING_VALIDATED TRUE CACHE INTERNAL "Threading validation completed")
  message(STATUS "XSigma: Threading support validated")
endif()

# Math library validation (cached)
if(NOT DEFINED TEMP_MATH_LIB_VALIDATED)
  if(NOT WIN32)
    check_library_exists(m sin "" HAVE_LIBM)
    if(NOT HAVE_LIBM)
      message(FATAL_ERROR "XSigma: Math library (libm) is required but not found")
    endif()
  endif()
  set(TEMP_MATH_LIB_VALIDATED TRUE CACHE INTERNAL "Math library validation completed")
  message(STATUS "XSigma: Math library support validated")
endif()

# =============================================================================
# Compiler
# Capability Validation (Cached)

include(CheckCXXSourceCompiles)

# C++17 features validation (cached).
# CMAKE_CXX_STANDARD is applied to targets later, and check_cxx_source_compiles
# does not inherit it. Clang's default language mode accepts structured
# bindings and if constexpr as extensions, but the MSVC STL hides std::optional
# until the TU is actually compiled as C++17. Probe with the project standard.
if(NOT DEFINED TEMP_CXX17_FEATURES_VALIDATED)
  set(_linalg_checks_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  if(DEFINED LINALG_CXX_STANDARD)
    set(_linalg_checks_cxx_standard "${LINALG_CXX_STANDARD}")
  elseif(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL "")
    set(_linalg_checks_cxx_standard "${CMAKE_CXX_STANDARD}")
  else()
    set(_linalg_checks_cxx_standard 17)
  endif()
  if(MSVC AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
                   AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
    string(APPEND CMAKE_REQUIRED_FLAGS
           " /Zc:__cplusplus /std:c++${_linalg_checks_cxx_standard}")
  else()
    string(APPEND CMAKE_REQUIRED_FLAGS " -std=c++${_linalg_checks_cxx_standard}")
  endif()

  # Test structured bindings
  check_cxx_source_compiles(
    "
        #include <tuple>
        int main() {
            auto [a, b] = std::make_tuple(1, 2);
            return a + b - 3;
        }
    "
    TEMP_HAS_STRUCTURED_BINDINGS
  )

  # Test if constexpr
  check_cxx_source_compiles(
    "
        constexpr int test_func(int x) {
            if constexpr (true) {
                return x * 2;
            } else {
                return x;
            }
        }
        int main() {
            constexpr int result = test_func(5);
            return result - 10;
        }
    "
    TEMP_HAS_IF_CONSTEXPR
  )

  # Test std::optional
  check_cxx_source_compiles(
    "
        #include <optional>
        int main() {
            std::optional<int> opt = 42;
            return opt.value_or(0) - 42;
        }
    "
    TEMP_HAS_STD_OPTIONAL
  )

  set(CMAKE_REQUIRED_FLAGS "${_linalg_checks_saved_required_flags}")
  unset(_linalg_checks_saved_required_flags)
  unset(_linalg_checks_cxx_standard)

  if(NOT TEMP_HAS_STRUCTURED_BINDINGS OR NOT TEMP_HAS_IF_CONSTEXPR OR NOT TEMP_HAS_STD_OPTIONAL)
    message(WARNING "XSigma: Compiler does not support required C++17 features")
  endif()

  set(TEMP_CXX17_FEATURES_VALIDATED TRUE CACHE INTERNAL "C++17 features validation completed")
  message(STATUS "XSigma: C++17 features validated")
endif()

# Exception handling validation (cached)
if(NOT DEFINED TEMP_EXCEPTION_HANDLING_VALIDATED)
  check_cxx_source_compiles(
    "
        #include <stdexcept>
        int main() {
            try {
                throw std::runtime_error(\"test\");
            } catch (const std::exception& e) {
                return 0;
            }
            return 1;
        }
    "
    TEMP_HAS_EXCEPTION_HANDLING
  )

  if(NOT TEMP_HAS_EXCEPTION_HANDLING)
    message(WARNING "XSigma: Exception handling not available - some features may be limited")
  endif()

  set(TEMP_EXCEPTION_HANDLING_VALIDATED TRUE CACHE INTERNAL
                                                   "Exception handling validation completed"
  )
  message(STATUS "XSigma: Exception handling validated")
endif()

# =============================================================================
# #
# Platform-Specific Validations

# Windows-specific checks
if(TEMP_PLATFORM_WINDOWS)
  if(NOT DEFINED TEMP_WINDOWS_VALIDATED)
    check_include_file("windows.h" HAVE_WINDOWS_H)
    if(NOT HAVE_WINDOWS_H)
      message(FATAL_ERROR "XSigma: Windows.h header not found on Windows platform")
    endif()
    set(TEMP_WINDOWS_VALIDATED TRUE CACHE INTERNAL "Windows validation completed")
    message(STATUS "XSigma: Windows platform validation completed")
  endif()
endif()

# Unix-specific checks
if(TEMP_PLATFORM_LINUX OR TEMP_PLATFORM_MACOS)
  if(NOT DEFINED TEMP_UNIX_VALIDATED)
    check_include_file("unistd.h" HAVE_UNISTD_H)
    check_include_file("pthread.h" HAVE_PTHREAD_H)
    if(NOT HAVE_UNISTD_H OR NOT HAVE_PTHREAD_H)
      message(FATAL_ERROR "XSigma: Required Unix headers not found")
    endif()
    set(TEMP_UNIX_VALIDATED TRUE CACHE INTERNAL "Unix validation completed")
    message(STATUS "XSigma: Unix platform validation completed")
  endif()
endif()

# =============================================================================
# Validation
# Completion and Caching

# Mark validation as completed for this configuration
set(TEMP_VALIDATION_COMPLETED "${TEMP_VALIDATION_CACHE_KEY}"
    CACHE INTERNAL "Validation completed for configuration"
)

# Store validation summary
set(TEMP_VALIDATION_SUMMARY
    "Platform: ${TEMP_PLATFORM}, Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, C++: ${TEMP_CXX_STANDARD}"
    CACHE INTERNAL "Validation summary"
)

message(STATUS "Checks: System validation completed successfully")
message(STATUS "Checks: ${TEMP_VALIDATION_SUMMARY}")

# =============================================================================
# End of
# checks.cmake
