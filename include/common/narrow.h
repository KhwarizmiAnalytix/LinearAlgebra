#pragma once

#include "include/logging.h"

#include <type_traits>

// LinearAlgebra — checked integer narrowing for the BLAS/LAPACKE and GPU
// (cuBLAS/cuSOLVER) call sites that pass linalg::index_t-sized sizes,
// strides, and counts into vendor APIs taking a narrower `int`. These sites
// used to narrow with a plain `static_cast<int>`, silently truncating any
// value that overflows `int` (see docs/backend_roadmap.md §2.1 item 2).
// `checked_narrow` keeps the same cast but reports overflow through
// LOGGING_THROW instead of truncating it.
namespace linalg
{
namespace detail
{

template <typename To, typename From>
To checked_narrow(From value)
{
    static_assert(std::is_integral<From>::value && std::is_integral<To>::value,
        "checked_narrow requires integral types");

    const auto narrowed = static_cast<To>(value);
    LOGGING_CHECK(static_cast<From>(narrowed) == value,
        "checked_narrow: value {} does not fit in target type", value);
    return narrowed;
}

}  // namespace detail
}  // namespace linalg
