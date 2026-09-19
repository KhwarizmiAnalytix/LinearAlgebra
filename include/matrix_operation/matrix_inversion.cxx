#include "include/matrix_operation/matrix_inversion.h"

#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/matrix_inversion_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

LINALG_DEFINE_DISPATCH(invert_f32_fn, invert_f32_stub);
LINALG_DEFINE_DISPATCH(invert_f64_fn, invert_f64_stub);

// matrix_determinant just reads the diagonal of an already-decomposed
// buffer — there is no vendor-library "determinant" routine to dispatch to,
// so this always runs directly against whatever lu_decomposition /
// cholesky_decomposition (themselves dispatched, default device_type::cpu)
// already wrote into `m`.
#define A(i, j) m[(i) * lda + (j)]

template <typename T>
T lu_determinant(T* m, quarisma_int lda)
{
    T det = m[0];
    for (quarisma_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

template <typename T>
T cholesky_determinant(T* m, quarisma_int lda)
{
    T det = m[0];
    for (quarisma_int i = 1; i < lda; ++i)
    {
        det *= A(i, i);
    }
    return det;
}

#undef A

template <typename T>
T matrix_determinant_helper(T* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    T ret = 0.;
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        {
            if (lu_decomposition(m, lda, pivot))
            {
                ret = lu_determinant(m, lda);
            }
        }
            return ret;

        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            return lu_determinant(m, lda);

        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        {
            if (cholesky_decomposition(m, lda, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR))
            {
                ret = cholesky_determinant(m, lda);
            }
        }
            return ret;

        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            return cholesky_determinant(m, lda);
        default:
            LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

}  // namespace detail

//-----------------------------------------------------------------------------
void matrix_invert(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type, device_type device)
{
    detail::invert_f32_stub.resolve(device)(m, pivot, lda, type);
}

//-----------------------------------------------------------------------------
void matrix_invert(
    double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type, device_type device)
{
    detail::invert_f64_stub.resolve(device)(m, pivot, lda, type);
}

//-----------------------------------------------------------------------------
float matrix_determinant(float* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_helper(m, pivot, lda, type);
}

//-----------------------------------------------------------------------------
double matrix_determinant(double* m, quarisma_int* pivot, quarisma_int lda, linear_solver_type type)
{
    return detail::matrix_determinant_helper(m, pivot, lda, type);
}
}  // namespace linalg
