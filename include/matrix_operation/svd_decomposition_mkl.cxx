#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

#include <algorithm>

#include "include/matrix_operation/svd_decomposition_dispatch.h"
#include "include/memory/allocator.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

void svd_mkl_f32(
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
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', rows, columns, U, ldu, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_sgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', rows, columns, VT, ldv, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

void svd_mkl_f64(
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
    if (rows > columns)
    {
        std::copy_n(A, rows * columns, U);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'O', 'S', rows, columns, U, ldu, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    else
    {
        std::copy_n(A, rows * columns, VT);
        auto info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'S', 'O', rows, columns, VT, ldv, S, U, ldu, VT, ldv, temp);  //NOLINT
        if (info != 0)
        {
            LINALG_THROW("mkl SVD was unsuccessful");
        }
    }
    Allocator::free(temp);
}

LINALG_REGISTER_DISPATCH(svd_f32_stub, mkl, svd_mkl_f32);
LINALG_REGISTER_DISPATCH(svd_f64_stub, mkl, svd_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
