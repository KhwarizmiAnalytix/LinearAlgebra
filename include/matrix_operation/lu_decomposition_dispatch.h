#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/lu_decomposition.h"

namespace linalg
{
namespace detail
{

using lu_f32_fn = bool (*)(float*, quarisma_int, quarisma_int*);
using lu_f64_fn = bool (*)(double*, quarisma_int, quarisma_int*);

LINALG_DECLARE_DISPATCH(lu_f32_fn, lu_f32_stub);
LINALG_DECLARE_DISPATCH(lu_f64_fn, lu_f64_stub);

}  // namespace detail
}  // namespace linalg
