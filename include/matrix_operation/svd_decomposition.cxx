#include "include/matrix_operation/svd_decomposition.h"

#include "include/matrix_operation/svd_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(svd_f32_fn, svd_f32_stub);
LINALG_DEFINE_DISPATCH(svd_f64_fn, svd_f64_stub);

}  // namespace detail

//-----------------------------------------------------------------------------
void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv,
    device_type device)
{
    detail::svd_f32_stub.resolve(device)(rows, columns, A, lda, S, U, ldu, VT, ldv);
}

//-----------------------------------------------------------------------------
void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv,
    device_type device)
{
    detail::svd_f64_stub.resolve(device)(rows, columns, A, lda, S, U, ldu, VT, ldv);
}
}  // namespace linalg
