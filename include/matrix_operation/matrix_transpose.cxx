#include "include/matrix_operation/matrix_transpose.h"

#include "include/matrix_operation/matrix_transpose_dispatch.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(transpose_f32_fn, transpose_f32_stub);
LINALG_DEFINE_DISPATCH(transpose_f64_fn, transpose_f64_stub);

}  // namespace detail

void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m, device_type device)
{
    detail::transpose_f32_stub.resolve(device)(rows, columns, m);
}

void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m, device_type device)
{
    detail::transpose_f64_stub.resolve(device)(rows, columns, m);
}

}  // namespace linalg
