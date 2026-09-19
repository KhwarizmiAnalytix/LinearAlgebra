#include "include/common/configure.h"  // IWYU pragma: keep

// LAPACKE-dependent — see include/common/configure.h.in and
// cholesky_decomposition_blas.cxx.
#if defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

#include <lapacke.h>

#include <algorithm>

#include "include/matrix_operation/svd_decomposition_dispatch.h"
#include "include/memory/allocator.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

void svd_blas_f32(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<float>;
    auto* temp      = Allocator::allocate(ldu - 1);
    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', r, c, U, static_cast<lapack_int>(ldu), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', r, c, VT, static_cast<lapack_int>(ldv), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

void svd_blas_f64(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv)
{
    using Allocator = linalg::allocator<double>;
    auto* temp      = Allocator::allocate(ldu - 1);
    const auto r = static_cast<lapack_int>(rows);
    const auto c = static_cast<lapack_int>(columns);
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', r, c, U, static_cast<lapack_int>(ldu), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', r, c, VT, static_cast<lapack_int>(ldv), S, U,
            static_cast<lapack_int>(ldu), VT, static_cast<lapack_int>(ldv), temp);
        if (info != 0)
        {
            LINALG_THROW("blas_lapack SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

LINALG_REGISTER_DISPATCH(svd_f32_stub, blas, svd_blas_f32);
LINALG_REGISTER_DISPATCH(svd_f64_stub, blas, svd_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE
