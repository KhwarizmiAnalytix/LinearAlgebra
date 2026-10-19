#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE

#include "include/matrix_operation/lu_decomposition.h"

#include "include/common/linalg_checks.h"

#include "include/common/backend.h"

// LAPACKE-dependent — see CMakeLists.txt and
// cholesky_decomposition.cpp: Apple's Accelerate framework has no LAPACKE
// wrapper, so this file only compiles when real lapacke.h exists.
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

#include "include/common/workspace_arena.h"

#include "include/common/linalg_profiler.h"

// BLAS/LAPACKE specializations for lu_decomposition.h (the native backend lives in
// lu_decomposition.cpp). The build compiles this file only when LINALG_ENABLE_BLAS is
// on; the #if below is kept as a second guard.
namespace linalg
{
// Same checks as lu_decomposition.cpp; each backend translation unit keeps its own
// internal-linkage copy so linking several backends cannot clash.
static void check_lu_inputs(const void* m, const void* pivot, linalg_int lda)
{
    LINALG_CHECK(
        m != nullptr && pivot != nullptr, "lu_decomposition::apply: m and pivot must not be null");
    LINALG_CHECK(lda > 0, "lu_decomposition::apply: lda must be positive", lda);
}

namespace
{
// lapack_int (a generic LAPACKE's pivot element type, typically 32-bit even
// on 64-bit platforms) does not necessarily match linalg_int (64-bit), so
// the caller's pivot buffer can't be passed straight through — unlike the
// mkl backend, which assumes an ILP64 MKL build where MKL_INT == linalg_int.
// `ipiv` (n lapack_int) is caller-owned scratch, carved from `workspace`.
size_t lu_blas_workspace_size(linalg_int lda)
{
    detail::workspace_arena arena;
    arena.next<lapack_int>(static_cast<size_t>(lda));
    return arena.total_size();
}
}  // namespace

template <>
bool lu_decomposition<float, linalg_backend::blas_lapacke>::apply(
    float* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size, void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/blas_lapacke<float>");
    check_lu_inputs(m, pivot, lda);
    LINALG_CHECK(workspace_size >= lu_blas_workspace_size(lda),
        "lu_decomposition: workspace_size too small");
    const auto              n = static_cast<lapack_int>(lda);
    detail::workspace_arena arena(workspace, workspace_size);
    lapack_int*             ipiv = arena.next<lapack_int>(static_cast<size_t>(n));
    const auto              info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, m, n, ipiv);
    for (lapack_int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<linalg_int>(ipiv[i]);
    }
    return info == 0;
}

template <>
bool lu_decomposition<double, linalg_backend::blas_lapacke>::apply(
    double* m, linalg_int lda, linalg_int* pivot, void* workspace, size_t workspace_size, void* /*stream*/)
{
    LINALG_SCOPE("lu_decomposition/blas_lapacke<double>");
    check_lu_inputs(m, pivot, lda);
    LINALG_CHECK(workspace_size >= lu_blas_workspace_size(lda),
        "lu_decomposition: workspace_size too small");
    const auto              n = static_cast<lapack_int>(lda);
    detail::workspace_arena arena(workspace, workspace_size);
    lapack_int*             ipiv = arena.next<lapack_int>(static_cast<size_t>(n));
    const auto              info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, m, n, ipiv);
    for (lapack_int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<linalg_int>(ipiv[i]);
    }
    return info == 0;
}

template <> size_t lu_decomposition<float, linalg_backend::blas_lapacke>::workspace_size(linalg_int lda)
{
    return lu_blas_workspace_size(lda);
}

template <> size_t lu_decomposition<double, linalg_backend::blas_lapacke>::workspace_size(linalg_int lda)
{
    return lu_blas_workspace_size(lda);
}

}  // namespace linalg
#endif  // LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
