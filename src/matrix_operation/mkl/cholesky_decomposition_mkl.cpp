#if LINALG_HAS_MKL

#include "include/matrix_operation/cholesky_decomposition.h"

#include "include/common/backend.h"
#include "include/common/linalg_checks.h"

#include <mkl.h>

// MKL specializations for cholesky_decomposition.h (the native backend lives in
// cholesky_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_MKL is
// on; the #if below is kept as a second guard.
namespace linalg
{

namespace detail
{

// Same checks as cholesky_decomposition.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_cholesky_inputs(const void* C, linalg_int lda)
{
    LINALG_CHECK(C != nullptr, "cholesky_decomposition::apply: C must not be null");
    LINALG_CHECK(lda > 0, "cholesky_decomposition::apply: lda must be positive", lda);
}

}  // namespace detail

template <>
bool cholesky_decomposition<float, linalg_backend::mkl>::apply(
    float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    detail::check_cholesky_inputs(C, lda);
    linalg_int n = lda;
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

template <>
bool cholesky_decomposition<double, linalg_backend::mkl>::apply(
    double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    detail::check_cholesky_inputs(C, lda);
    linalg_int n = lda;
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

// apply_aad has no vendor-library equivalent: forward to the native
// scalar adjoint defined in cholesky_decomposition.cpp.
template <>
bool cholesky_decomposition<float, linalg_backend::mkl>::apply_aad(
    float* C_aad, const float* C, linalg_int lda, cholesky_decomposition_enum type, float* A_aad)
{
    return cholesky_decomposition<float, linalg_backend::native>::apply_aad(C_aad, C, lda, type, A_aad);
}

template <>
bool cholesky_decomposition<double, linalg_backend::mkl>::apply_aad(
    double* C_aad, const double* C, linalg_int lda, cholesky_decomposition_enum type, double* A_aad)
{
    return cholesky_decomposition<double, linalg_backend::native>::apply_aad(C_aad, C, lda, type, A_aad);
}

}  // namespace linalg
#endif  // LINALG_HAS_MKL
