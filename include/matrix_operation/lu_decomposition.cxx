#include "include/matrix_operation/lu_decomposition.h"

#include "include/matrix_operation/lu_decomposition_dispatch.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(lu_f32_fn, lu_f32_stub);
LINALG_DEFINE_DISPATCH(lu_f64_fn, lu_f64_stub);

}  // namespace detail

bool lu_decomposition(float* m, quarisma_int lda, quarisma_int* pivot, device_type device)
{
    return detail::lu_f32_stub.resolve(device)(m, lda, pivot);
}

bool lu_decomposition(double* m, quarisma_int lda, quarisma_int* pivot, device_type device)
{
    return detail::lu_f64_stub.resolve(device)(m, lda, pivot);
}

}  // namespace linalg
