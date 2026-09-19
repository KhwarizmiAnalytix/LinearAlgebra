#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

#include "include/matrix_operation/cholesky_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{

bool cholesky_mkl_f32(float* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    quarisma_int n = lda;
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

bool cholesky_mkl_f64(double* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    quarisma_int n = lda;
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

LINALG_REGISTER_DISPATCH(cholesky_f32_stub, mkl, cholesky_mkl_f32);
LINALG_REGISTER_DISPATCH(cholesky_f64_stub, mkl, cholesky_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
