#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/linear_solver.h"

namespace linalg
{
namespace detail
{

using solver_f32_fn = void (*)(float*, quarisma_int*, quarisma_int, float*, linear_solver_type);
using solver_f64_fn = void (*)(double*, quarisma_int*, quarisma_int, double*, linear_solver_type);

LINALG_DECLARE_DISPATCH(solver_f32_fn, solver_f32_stub);
LINALG_DECLARE_DISPATCH(solver_f64_fn, solver_f64_stub);

}  // namespace detail
}  // namespace linalg
