#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_MKL

#include <mkl.h>

#include "include/matrix_operation/matrix_transpose_dispatch.h"

namespace linalg
{
namespace detail
{

void transpose_mkl_f32(quarisma_long rows, quarisma_long columns, float* m)
{
    mkl_simatcopy('R', 'T', rows, columns, 1.F, m, columns, rows);
}

void transpose_mkl_f64(quarisma_long rows, quarisma_long columns, double* m)
{
    mkl_dimatcopy('R', 'T', rows, columns, 1., m, columns, rows);
}

LINALG_REGISTER_DISPATCH(transpose_f32_stub, mkl, transpose_mkl_f32);
LINALG_REGISTER_DISPATCH(transpose_f64_stub, mkl, transpose_mkl_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_MKL
