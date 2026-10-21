cmake_minimum_required(VERSION 3.24)
project(LoggingPackageFixture LANGUAGES C CXX)
if(NOT LINALG_SOURCE_DIR)
    message(FATAL_ERROR "LINALG_SOURCE_DIR must point to the checkout")
endif()
set(LINALG_ENABLE_EXTERNAL OFF)
set(XSIGMA_ENABLE_EXTERNAL OFF)
set(LOGGING_ENABLE_TESTING OFF)
set(LOGGING_ENABLE_BENCHMARK OFF)
set(LOGGING_ENABLE_EXAMPLES OFF)
include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")
linalg_add_dependency_subdirectory(
    "${LINALG_SOURCE_DIR}/ThirdParty/Logging" "${CMAKE_CURRENT_BINARY_DIR}/logging"
)
