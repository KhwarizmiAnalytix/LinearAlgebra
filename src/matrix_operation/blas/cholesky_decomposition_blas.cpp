#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/cholesky_decomposition.h"

#include "include/common/backend.h"
#include "include/common/narrow.h"
#include "include/common/linalg_checks.h"

// LAPACKE-dependent (see CMakeLists.txt): Apple's Accelerate framework ships
// CBLAS but not the LAPACKE row-major C wrapper, so this block only compiles
// when a real lapacke.h was found.
#include <lapacke.h>

#include "include/common/linalg_profiler.h"

// BLAS/LAPACKE specializations for cholesky_decomposition.h (the native backend lives in
// cholesky_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
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

// LAPACKE_?potrf only validates its uplo argument by printing to stderr (xerbla) and
// returning a nonzero info, not by throwing; reject an invalid enum here so this backend
// throws the same way cholesky_decomposition.cpp's native specializations do.
static void check_cholesky_type(cholesky_decomposition_enum type)
{
    LINALG_CHECK(type == cholesky_decomposition_enum::LOWER_TRIANGULAR ||
            type == cholesky_decomposition_enum::UPPER_TRIANGULAR,
        "cholesky_decomposition::apply: Unsupported enum type!");
}

}  // namespace detail

template <>
bool cholesky_decomposition<float, linalg_backend::blas_lapacke>::apply(
    float* C, linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_SCOPE("cholesky_decomposition/blas_lapacke<float>");
    detail::check_cholesky_inputs(C, lda);
    detail::check_cholesky_type(type);
    const auto n = detail::checked_narrow<int>(lda);
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

template <>
bool cholesky_decomposition<double, linalg_backend::blas_lapacke>::apply(
    double* C, linalg_int lda, cholesky_decomposition_enum type)
{
    LINALG_SCOPE("cholesky_decomposition/blas_lapacke<double>");
    detail::check_cholesky_inputs(C, lda);
    detail::check_cholesky_type(type);
    const auto n = detail::checked_narrow<int>(lda);
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n) == 0;
}

// apply_aad has no vendor-library equivalent: forward to the native
// scalar adjoint defined in cholesky_decomposition.cpp.
template <>
bool cholesky_decomposition<float, linalg_backend::blas_lapacke>::apply_aad(
    float* C_aad, const float* C, linalg_int lda, cholesky_decomposition_enum type, float* A_aad)
{
    return cholesky_decomposition<float, linalg_backend::native>::apply_aad(C_aad, C, lda, type, A_aad);
}

template <>
bool cholesky_decomposition<double, linalg_backend::blas_lapacke>::apply_aad(
    double* C_aad, const double* C, linalg_int lda, cholesky_decomposition_enum type, double* A_aad)
{
    return cholesky_decomposition<double, linalg_backend::native>::apply_aad(C_aad, C, lda, type, A_aad);
}
}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
