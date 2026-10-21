#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/cholesky_decomposition.h"

#include "include/common/backend.h"
#include "include/common/narrow.h"
#include "include/common/linalg_checks.h"

// LAPACKE-dependent (see CMakeLists.txt): Apple's Accelerate framework ships
// CBLAS but not the LAPACKE row-major C wrapper, so this block only compiles
// when a real lapacke.h was found.
#include <complex>   // needed by OpenBLAS lapack.h on MSVC (no _Complex support)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4190) // C-linkage returning std::complex<> (OpenBLAS lapacke.h)
#endif
#include <lapacke.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "blas_runtime.h"

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
bool cholesky_decomposition<float, linalg_backend::BLAS_LAPACKE>::apply(float* C,
    linalg_int                                                                lda,
    cholesky_decomposition_enum                                               type,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    int* info,
    void* /*stream*/)
{
    LINALG_SCOPE("cholesky_decomposition/blas_lapacke<float>");
    detail::check_cholesky_inputs(C, lda);
    detail::check_cholesky_type(type);
    const auto n       = detail::checked_narrow<int>(lda);
    const int  lapacke_info =
        LINALG_BLAS_FUNCTION(LAPACKE_spotrf)(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n);
    if (info != nullptr)
    {
        *info = lapacke_info;
    }
    return lapacke_info == 0;
}

template <>
bool cholesky_decomposition<double, linalg_backend::BLAS_LAPACKE>::apply(double* C,
    linalg_int                                                                 lda,
    cholesky_decomposition_enum                                                type,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    int* info,
    void* /*stream*/)
{
    LINALG_SCOPE("cholesky_decomposition/blas_lapacke<double>");
    detail::check_cholesky_inputs(C, lda);
    detail::check_cholesky_type(type);
    const auto n       = detail::checked_narrow<int>(lda);
    const int  lapacke_info =
        LINALG_BLAS_FUNCTION(LAPACKE_dpotrf)(LAPACK_ROW_MAJOR, static_cast<char>(type), n, C, n);
    if (info != nullptr)
    {
        *info = lapacke_info;
    }
    return lapacke_info == 0;
}

template <>
size_t cholesky_decomposition<float, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_int /*lda*/, cholesky_decomposition_enum /*type*/)
{
    return 0;
}

template <>
size_t cholesky_decomposition<double, linalg_backend::BLAS_LAPACKE>::workspace_size(
    linalg_int /*lda*/, cholesky_decomposition_enum /*type*/)
{
    return 0;
}

// apply_aad has no vendor-library equivalent: forward to the native
// scalar adjoint defined in cholesky_decomposition.cpp.
template <>
bool cholesky_decomposition<float, linalg_backend::BLAS_LAPACKE>::apply_aad(
    float* C_aad, const float* C, linalg_int lda, cholesky_decomposition_enum type, float* A_aad)
{
    return cholesky_decomposition<float, linalg_backend::NATIVE>::apply_aad(C_aad, C, lda, type, A_aad);
}

template <>
bool cholesky_decomposition<double, linalg_backend::BLAS_LAPACKE>::apply_aad(
    double* C_aad, const double* C, linalg_int lda, cholesky_decomposition_enum type, double* A_aad)
{
    return cholesky_decomposition<double, linalg_backend::NATIVE>::apply_aad(C_aad, C, lda, type, A_aad);
}
}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
