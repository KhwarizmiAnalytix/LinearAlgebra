#include "include/common/configure.h"  // IWYU pragma: keep

// LAPACKE-dependent (see include/common/configure.h.in): Apple's Accelerate
// framework ships CBLAS but not the LAPACKE row-major C wrapper, so this
// backend is only registered when a real lapacke.h was found; otherwise
// dispatch_stub::resolve() falls back to the scalar implementation.
#if defined(LINALG_ENABLE_BLAS) && defined(LINALG_BLAS_HAS_LAPACKE)

#include <lapacke.h>

#include "include/matrix_operation/cholesky_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{

bool cholesky_blas_f32(float* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    const auto n = static_cast<int>(lda);
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

bool cholesky_blas_f64(double* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    const auto n = static_cast<int>(lda);
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

LINALG_REGISTER_DISPATCH(cholesky_f32_stub, blas, cholesky_blas_f32);
LINALG_REGISTER_DISPATCH(cholesky_f64_stub, blas, cholesky_blas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_BLAS && LINALG_BLAS_HAS_LAPACKE
