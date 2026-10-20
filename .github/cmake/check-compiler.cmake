# Included after project(LinearAlgebra) to catch accidental toolchain fallback in CI.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL LINALG_CI_COMPILER_ID)
    message(FATAL_ERROR "CI expected ${LINALG_CI_COMPILER_ID}, found ${CMAKE_CXX_COMPILER_ID}")
endif()

string(REPLACE "." "\\." _linalg_ci_version_pattern "${LINALG_CI_COMPILER_VERSION}")
if(NOT CMAKE_CXX_COMPILER_VERSION MATCHES "^${_linalg_ci_version_pattern}(\\.|$)")
    message(FATAL_ERROR "CI expected ${LINALG_CI_COMPILER_ID} ${LINALG_CI_COMPILER_VERSION}.x, "
                        "found ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()
unset(_linalg_ci_version_pattern)

message(STATUS "CI verified ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
