#include "include/matrix_operation/linear_solver.h"

#include "include/matrix_operation/linear_solver_dispatch.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(solver_f32_fn, solver_f32_stub);
LINALG_DEFINE_DISPATCH(solver_f64_fn, solver_f64_stub);

}  // namespace detail

//-----------------------------------------------------------------------------
void linear_solver(
    float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type, device_type device)
{
    detail::solver_f32_stub.resolve(device)(m, pivot, lda, x, type);
}

//-----------------------------------------------------------------------------
void linear_solver(
    double*             m,
    quarisma_int*         pivot,
    quarisma_int          lda,
    double*             x,
    linear_solver_type type,
    device_type         device)
{
    detail::solver_f64_stub.resolve(device)(m, pivot, lda, x, type);
}
}  // namespace linalg
