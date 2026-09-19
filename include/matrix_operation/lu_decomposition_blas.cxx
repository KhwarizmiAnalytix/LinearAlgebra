#include "include/common/configure.h"  // IWYU pragma: keep

// LAPACKE-dependent — see include/common/configure.h.in and
// cholesky_decomposition_blas.cxx: Apple's Accelerate framework has no
// LAPACKE wrapper, so this backend only registers when real lapacke.h exists.
#if defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

#include <lapacke.h>

#include <vector>

#include "include/matrix_operation/lu_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{
namespace
{

// lapack_int (a generic LAPACKE's pivot element type, typically 32-bit even
// on 64-bit platforms) does not necessarily match quarisma_int (64-bit), so
// the caller's pivot buffer can't be passed straight through — unlike the
// mkl backend, which assumes an ILP64 MKL build where MKL_INT == quarisma_int.
bool lu_blas_impl(
    lapack_int (*getrf)(int, lapack_int, lapack_int, void*, lapack_int, lapack_int*),
    void* m,
    quarisma_int lda,
    quarisma_int* pivot)
{
    const auto           n = static_cast<lapack_int>(lda);
    std::vector<lapack_int> ipiv(static_cast<size_t>(n));
    const auto info = getrf(LAPACK_ROW_MAJOR, n, n, m, n, ipiv.data());
    for (lapack_int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<quarisma_int>(ipiv[static_cast<size_t>(i)]);
    }
    return info == 0;
}

}  // namespace

bool lu_blas_f32(float* m, quarisma_int lda, quarisma_int* pivot)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip) {
            return LAPACKE_sgetrf(order, r, c, static_cast<float*>(a), ld, ip);
        },
        m,
        lda,
        pivot);
}

bool lu_blas_f64(double* m, quarisma_int lda, quarisma_int* pivot)
{
    return lu_blas_impl(
        [](int order, lapack_int r, lapack_int c, void* a, lapack_int ld, lapack_int* ip) {
            return LAPACKE_dgetrf(order, r, c, static_cast<double*>(a), ld, ip);
        },
        m,
        lda,
        pivot);
}

LINALG_REGISTER_DISPATCH(lu_f32_stub, blas, lu_blas_f32);
LINALG_REGISTER_DISPATCH(lu_f64_stub, blas, lu_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE
