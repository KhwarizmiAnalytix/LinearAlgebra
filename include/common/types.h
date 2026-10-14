#pragma once

#include <cstdint>

// LinearAlgebra — shared index type for public API signatures (sizes,
// strides/leading dimensions, pivot/workspace counts).
//
// `linalg_int` / `linalg_long` used to be redefined independently in every
// public header (see git history pre-dating this file) as, respectively,
// a 64-bit signed and a 64-bit unsigned integer (MSVC: __int64 / unsigned
// __int64; everywhere else: long long int / unsigned long long int). They
// are kept here, as deprecated macro aliases with their original types, so
// existing call sites and out-of-tree consumers keep compiling unchanged;
// new code should spell out linalg::index_t. Remove both macros at 2.0 —
// see docs/backend_roadmap.md §2.2 on unifying the signed/unsigned split.
namespace linalg
{
using index_t = std::int64_t;
}  // namespace linalg

#if !defined(linalg_int)
#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define linalg_int __int64
#define linalg_long unsigned __int64
#else
#define linalg_int long long int
#define linalg_long unsigned long long int
#endif
#endif
