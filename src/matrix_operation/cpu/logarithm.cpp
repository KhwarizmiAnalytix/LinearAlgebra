#include "include/matrix_operation/logarithm.h"

#include "include/common/linalg_profiler.h"
#include <cmath>
#include <limits>

#include "src/common/logarithm_core.h"
#include "src/matrix_operation/logarithm_workspace.h"

namespace linalg
{
namespace
{
template <typename T, linalg_backend Backend>
void logarithm_impl(const T* a,
    linalg_int               n,
    linalg_int               lda,
    T*                       result,
    linalg_int               ldc,
    void*                    workspace,
    size_t                   workspace_size)
{
    detail::check_logarithm_order(n);
    detail::check_schur_function_arguments<T, Backend>(a,
        n,
        lda,
        result,
        ldc,
        workspace,
        workspace_size,
        detail::logarithm_workspace_size<T, Backend>(n));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p       = detail::logarithm_plan<T, Backend>(arena, n);
    const T                 epsilon = std::numeric_limits<T>::epsilon();

    schur_decomposition<T, Backend>::apply(a,
        n,
        lda,
        p.schur.sr,
        p.schur.si,
        n,
        p.schur.qr,
        p.schur.qi,
        n,
        p.schur.schur_workspace,
        p.schur.schur_workspace_bytes);

    linalg_schur::logarithm_schur_prepare(
        p.schur.sr, p.schur.si, static_cast<int>(n), p.h, p.f, p.diagonal, p.state, epsilon);
    LINALG_CHECK(p.state[0] > T(0),
        "matrix {}: nonfinite input, unsupported real domain, or numerical failure",
        "logarithm");
    for (int node = 0; node < 7; ++node)
    {
        linalg_schur::pack_logarithm_solve(
            p.h, p.state, p.triangular_a, p.triangular_b, static_cast<int>(n), node);
        triangular_solver<T, Backend>::apply(true,
            false,
            true,
            2 * n,
            n,
            p.triangular_a,
            2 * n,
            p.triangular_b,
            n,
            p.triangular_x,
            n,
            p.triangular_workspace,
            p.triangular_bytes);
        linalg_schur::accumulate_logarithm_solve(
            p.f, p.state, p.triangular_x, static_cast<int>(n), node);
    }
    linalg_schur::logarithm_schur_finish(
        p.f, p.state, p.diagonal, static_cast<int>(n), p.schur.fr, p.schur.fi);

    detail::reconstruct_schur_function<T, Backend>(p.schur,
        n,
        nullptr,
        [n](const T* x, const T* y, T* out, T sign)
        { linalg_schur::combine(x, y, out, static_cast<int>(n * n), sign); });
    const bool published = linalg_schur::publish(
        p.schur.fr, p.schur.fi, static_cast<int>(n), result, static_cast<int>(ldc), epsilon);
    LINALG_CHECK(published, "matrix {}: nonfinite result or unsupported real domain", "logarithm");
}

}  // namespace

template <typename T, linalg_backend Backend>
void logarithm<T, Backend>::apply(const T* a,
    linalg_int                             n,
    linalg_int                             lda,
    T*                                     result,
    linalg_int                             ldc,
    void*                                  workspace,
    size_t                                 workspace_size,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    logarithm_impl<T, Backend>(a, n, lda, result, ldc, workspace, workspace_size);
}

template <typename T, linalg_backend Backend>
size_t logarithm<T, Backend>::workspace_size(linalg_int n)
{
    detail::check_logarithm_order(n);
    return detail::logarithm_workspace_size<T, Backend>(n);
}

template struct logarithm<float, linalg_backend::NATIVE>;
template struct logarithm<double, linalg_backend::NATIVE>;
#if LINALG_HAS_BLAS
template struct logarithm<float, linalg_backend::BLAS_LAPACKE>;
template struct logarithm<double, linalg_backend::BLAS_LAPACKE>;
#endif
#if LINALG_HAS_MKL
template struct logarithm<float, linalg_backend::MKL>;
template struct logarithm<double, linalg_backend::MKL>;
#endif
}  // namespace linalg
