cmake_minimum_required(VERSION 3.24)
project(ScalarsPackageFixture LANGUAGES C CXX)
if(NOT LINALG_SOURCE_DIR)
    message(FATAL_ERROR "LINALG_SOURCE_DIR must point to the checkout")
endif()
include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")
linalg_add_dependency_subdirectory(
    "${LINALG_SOURCE_DIR}/ThirdParty/Scalars" "${CMAKE_CURRENT_BINARY_DIR}/scalars"
)
