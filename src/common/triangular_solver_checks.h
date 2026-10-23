#pragma once

#include <limits>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/common/types.h"

namespace linalg::detail
{
inline void check_triangular_dimensions(linalg_int n, linalg_int nrhs)
{
    LINALG_CHECK(n > 0 && nrhs > 0, "triangular_solver: dimensions must be positive");
    checked_narrow<int>(n);
    checked_narrow<int>(nrhs);
}

inline void check_triangular_arguments(const void* a,
    const void*                                    b,
    const void*                                    x,
    linalg_int                                     n,
    linalg_int                                     nrhs,
    linalg_int                                     lda,
    linalg_int                                     ldb,
    linalg_int                                     ldx)
{
    check_triangular_dimensions(n, nrhs);
    LINALG_CHECK(a && b && x, "triangular_solver: matrices must not be null");
    LINALG_CHECK(
        a != b && a != x && b != x, "triangular_solver: matrices must use distinct buffers");
    LINALG_CHECK(lda >= n && ldb >= nrhs && ldx >= nrhs, "triangular_solver: invalid row stride");
    const auto limit = std::numeric_limits<linalg_int>::max() / n;
    LINALG_CHECK(lda <= limit && ldb <= limit && ldx <= limit,
        "triangular_solver: row stride span exceeds supported range");
}
}  // namespace linalg::detail
