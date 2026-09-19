#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/svd_decomposition.h"

namespace linalg
{
namespace detail
{

using svd_f32_fn = void (*)(
    quarisma_long, quarisma_long, float*, quarisma_long, float*, float*, quarisma_long, float*, quarisma_long);
using svd_f64_fn = void (*)(
    quarisma_long,
    quarisma_long,
    double*,
    quarisma_long,
    double*,
    double*,
    quarisma_long,
    double*,
    quarisma_long);

LINALG_DECLARE_DISPATCH(svd_f32_fn, svd_f32_stub);
LINALG_DECLARE_DISPATCH(svd_f64_fn, svd_f64_stub);

}  // namespace detail
}  // namespace linalg
