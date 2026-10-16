/*
 * LinearAlgebra — LINALG_* aliases for the LOGGING_* check/throw macros, so
 * first-party code spells error handling in this project's own macro
 * namespace instead of reaching into Logging directly.
 */
#pragma once

#include "include/util/exception.h"

#define LINALG_CHECK(cond, ...) LOGGING_CHECK(cond, ##__VA_ARGS__)
#define LINALG_THROW(format_str, ...) LOGGING_THROW(format_str, ##__VA_ARGS__)
