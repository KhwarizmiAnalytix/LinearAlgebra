#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/matrix_transpose.h"

namespace linalg
{
namespace detail
{

using transpose_f32_fn = void (*)(quarisma_long, quarisma_long, float*);
using transpose_f64_fn = void (*)(quarisma_long, quarisma_long, double*);

LINALG_DECLARE_DISPATCH(transpose_f32_fn, transpose_f32_stub);
LINALG_DECLARE_DISPATCH(transpose_f64_fn, transpose_f64_stub);

}  // namespace detail
}  // namespace linalg
