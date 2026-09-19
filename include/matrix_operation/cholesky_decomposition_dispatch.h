#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/cholesky_decomposition.h"

namespace linalg
{
namespace detail
{

using cholesky_f32_fn = bool (*)(float*, quarisma_int, cholesky_decomposition_enum);
using cholesky_f64_fn = bool (*)(double*, quarisma_int, cholesky_decomposition_enum);

LINALG_DECLARE_DISPATCH(cholesky_f32_fn, cholesky_f32_stub);
LINALG_DECLARE_DISPATCH(cholesky_f64_fn, cholesky_f64_stub);

}  // namespace detail
}  // namespace linalg
