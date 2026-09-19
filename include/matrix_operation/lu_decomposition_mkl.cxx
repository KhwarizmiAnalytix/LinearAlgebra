#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl_lapacke.h>

#include "include/matrix_operation/lu_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{

bool lu_mkl_f32(float* m, quarisma_int lda, quarisma_int* pivot)
{
    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

bool lu_mkl_f64(double* m, quarisma_int lda, quarisma_int* pivot)
{
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, lda, lda, m, lda, pivot) == 0;
}

LINALG_REGISTER_DISPATCH(lu_f32_stub, mkl, lu_mkl_f32);
LINALG_REGISTER_DISPATCH(lu_f64_stub, mkl, lu_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
