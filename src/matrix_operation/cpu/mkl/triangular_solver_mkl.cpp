#if LINALG_HAS_MKL
#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/triangular_solver.h"
#include "src/common/triangular_solver_checks.h"
#include <algorithm>
#include <mkl.h>

namespace linalg
{

template <>
void triangular_solver<float, linalg_backend::MKL>::apply(bool upper,
    bool                                                       transpose,
    bool                                                       unit,
    linalg_int                                                 n,
    linalg_int                                                 nrhs,
    const float*                                               a,
    linalg_int                                                 lda,
    const float*                                               b,
    linalg_int                                                 ldb,
    float*                                                     x,
    linalg_int                                                 ldx,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    for (linalg_int i = 0; i < n; ++i)
        std::copy_n(b + i * ldb, nrhs, x + i * ldx);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_strsm(CblasRowMajor,
        CblasLeft,
        upper ? CblasUpper : CblasLower,
        transpose ? CblasTrans : CblasNoTrans,
        unit ? CblasUnit : CblasNonUnit,
        detail::checked_narrow<MKL_INT>(n),
        detail::checked_narrow<MKL_INT>(nrhs),
        float(1),
        a,
        detail::checked_narrow<MKL_INT>(lda),
        x,
        detail::checked_narrow<MKL_INT>(ldx));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template <>
size_t triangular_solver<float, linalg_backend::MKL>::workspace_size(linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    return 0;
}

template <>
void triangular_solver<double, linalg_backend::MKL>::apply(bool upper,
    bool                                                        transpose,
    bool                                                        unit,
    linalg_int                                                  n,
    linalg_int                                                  nrhs,
    const double*                                               a,
    linalg_int                                                  lda,
    const double*                                               b,
    linalg_int                                                  ldb,
    double*                                                     x,
    linalg_int                                                  ldx,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    for (linalg_int i = 0; i < n; ++i)
        std::copy_n(b + i * ldb, nrhs, x + i * ldx);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_dtrsm(CblasRowMajor,
        CblasLeft,
        upper ? CblasUpper : CblasLower,
        transpose ? CblasTrans : CblasNoTrans,
        unit ? CblasUnit : CblasNonUnit,
        detail::checked_narrow<MKL_INT>(n),
        detail::checked_narrow<MKL_INT>(nrhs),
        double(1),
        a,
        detail::checked_narrow<MKL_INT>(lda),
        x,
        detail::checked_narrow<MKL_INT>(ldx));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

template <>
size_t triangular_solver<double, linalg_backend::MKL>::workspace_size(linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    return 0;
}
}  // namespace linalg
#endif  // LINALG_HAS_MKL
