#pragma once

#include "include/common/dispatch_stub.h"
#include "include/matrix_operation/matrix_multiplication.h"

namespace linalg
{
namespace detail
{

using matmul_f32_fn = void (*)(
    bool,
    bool,
    quarisma_int,
    quarisma_int,
    quarisma_int,
    const float*,
    quarisma_int,
    const float*,
    quarisma_int,
    float*,
    quarisma_int);

using matmul_f64_fn = void (*)(
    bool,
    bool,
    quarisma_int,
    quarisma_int,
    quarisma_int,
    const double*,
    quarisma_int,
    const double*,
    quarisma_int,
    double*,
    quarisma_int);

LINALG_DECLARE_DISPATCH(matmul_f32_fn, matmul_f32_stub);
LINALG_DECLARE_DISPATCH(matmul_f64_fn, matmul_f64_stub);

}  // namespace detail
}  // namespace linalg
