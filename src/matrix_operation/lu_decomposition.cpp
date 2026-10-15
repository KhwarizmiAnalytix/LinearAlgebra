#include "include/matrix_operation/lu_decomposition.h"

#include <include/logging.h>

#include "include/common/backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "include/common/macros.h"

namespace linalg
{

// Null/positivity checks shared by every lu_backend<T, Backend>::apply
// below, so callers (linear_solver.cpp, matrix_inversion.cpp, tests) get the
// same validation regardless of which backend is active. There is no
// non-detail lu_decomposition() wrapper to put this in any more — callers
// use lu_backend<T, kActiveBackend>::apply directly.
static void check_lu_inputs(const void* m, const void* pivot, linalg_int lda)
{
    LOGGING_CHECK(
        m != nullptr && pivot != nullptr, "lu_backend::apply: m and pivot must not be null");
    LOGGING_CHECK(lda > 0, "lu_backend::apply: lda must be positive", lda);
}

// Only the native specializations are defined here. The mkl and blas_lapacke
// specializations live in mkl/lu_decomposition_mkl.cpp and
// blas/lu_decomposition_blas.cpp, which the build compiles only when
// LINALG_ENABLE_MKL / LINALG_ENABLE_BLAS is on.

namespace
{

#define A(i, j) m[(i) * lda + (j)]

template <typename T>
bool lu_decomposition_scalar_impl(
    T* m, linalg_long lda, LINALG_UNUSED linalg_int* pivot, LINALG_UNUSED T tolerance)
{
    using value_t   = T;
    using size_type = linalg_long;

#if LINALG_HAS_LU_PIVOTING
    // Initialize pivot array. `pivot` holds exactly `lda` entries (LAPACK's
    // ipiv convention: pivot[i] is the 1-based row swapped with row i) —
    // pivotCount (the swap count, used only to flip the determinant's sign
    // under odd permutations) is tracked locally rather than as a trailing
    // pivot[lda], which used to overflow the caller's n-sized buffer by one
    // element (see docs/backend_roadmap.md §2.1 item 1).
    for (size_type i = 0; i < lda; ++i)
    {
        pivot[i] = static_cast<linalg_int>(i + 1);
    }
    linalg_int pivotCount = 0;
#endif
    for (size_type i = 0; i < lda; ++i)
    {
#if LINALG_HAS_LU_PIVOTING
        value_t   maxA = std::abs(A(i, i));
        size_type imax = i;
        for (size_type k = i + 1; k < lda; ++k)
        {
            const auto absA = std::fabs(A(k, i));
            if (absA > maxA)
            {
                maxA = absA;
                imax = k;
            }
        }
        if (maxA < tolerance)
        {
            return false;
        }
        if (imax != i)
        {
            pivot[i] = pivot[imax];
            for (size_type j = 0; j < lda; ++j)
            {
                auto& ptri = A(i, j);
                auto& ptrj = A(imax, j);
                std::swap(ptri, ptrj);
            }
            pivotCount++;
        }
#endif
        auto norm = 1. / A(i, i);
        auto size = lda - i - 1;

        const auto* const l_i = &A(i, i + 1);
        for (size_type j = i + 1; j < lda; ++j)
        {
            auto* l_j = &A(j, i);
            l_j[0] *= static_cast<T>(norm);
            const auto a_ji = l_j[0];
            l_j++;
            for (size_type k = 0; k < size; ++k)
            {
                l_j[k] -= a_ji * l_i[k];
            }
        }
    }
    return true;
}

#undef A

}  // namespace

template <>
bool lu_backend<float, linalg_backend::native>::apply(
    float* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<float>::epsilon());
}

template <>
bool lu_backend<double, linalg_backend::native>::apply(
    double* m, linalg_int lda, linalg_int* pivot, void* /*workspace*/, size_t /*workspace_size*/)
{
    check_lu_inputs(m, pivot, lda);
    return lu_decomposition_scalar_impl(m, lda, pivot, std::numeric_limits<double>::epsilon());
}

template <> size_t lu_backend<float, linalg_backend::native>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}

template <> size_t lu_backend<double, linalg_backend::native>::workspace_size(linalg_int /*lda*/)
{
    return 0;
}
}  // namespace linalg
