#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/matrix_inversion.h"

namespace linalg
{
namespace detail
{

using invert_f32_fn = void (*)(float*, quarisma_int*, quarisma_int, linear_solver_type);
using invert_f64_fn = void (*)(double*, quarisma_int*, quarisma_int, linear_solver_type);

LINALG_DECLARE_DISPATCH(invert_f32_fn, invert_f32_stub);
LINALG_DECLARE_DISPATCH(invert_f64_fn, invert_f64_stub);

}  // namespace detail
}  // namespace linalg
