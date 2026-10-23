#include "include/matrix_operation/square_root.h"

#include "include/common/linalg_profiler.h"
#include <cmath>
#include <limits>

#include "src/common/square_root_core.h"
#include "src/matrix_operation/square_root_workspace.h"

namespace linalg
{
namespace
{
template <typename T, linalg_backend Backend>
void square_root_impl(const T* a,
    linalg_int                 n,
    linalg_int                 lda,
    T*                         result,
    linalg_int                 ldc,
    void*                      workspace,
    size_t                     workspace_size)
{
    detail::check_schur_function_arguments<T, Backend>(a,
        n,
        lda,
        result,
        ldc,
        workspace,
        workspace_size,
        detail::square_root_workspace_size<T, Backend>(n));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p       = detail::square_root_plan<T, Backend>(arena, n);
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

    const bool success = linalg_schur::square_root_schur(
        p.schur.sr, p.schur.si, static_cast<int>(n), p.h, p.schur.fr, p.schur.fi, epsilon);
    LINALG_CHECK(success,
        "matrix {}: nonfinite input, unsupported real domain, or numerical failure",
        "square_root");

    detail::reconstruct_schur_function<T, Backend>(p.schur,
        n,
        nullptr,
        [n](const T* x, const T* y, T* out, T sign)
        { linalg_schur::combine(x, y, out, static_cast<int>(n * n), sign); });
    const bool published = linalg_schur::publish(
        p.schur.fr, p.schur.fi, static_cast<int>(n), result, static_cast<int>(ldc), epsilon);
    LINALG_CHECK(
        published, "matrix {}: nonfinite result or unsupported real domain", "square_root");
}

}  // namespace

template <typename T, linalg_backend Backend>
void square_root<T, Backend>::apply(const T* a,
    linalg_int                               n,
    linalg_int                               lda,
    T*                                       result,
    linalg_int                               ldc,
    void*                                    workspace,
    size_t                                   workspace_size,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    square_root_impl<T, Backend>(a, n, lda, result, ldc, workspace, workspace_size);
}

template <typename T, linalg_backend Backend>
size_t square_root<T, Backend>::workspace_size(linalg_int n)
{
    return detail::square_root_workspace_size<T, Backend>(n);
}

template struct square_root<float, linalg_backend::NATIVE>;
template struct square_root<double, linalg_backend::NATIVE>;
#if LINALG_HAS_BLAS
template struct square_root<float, linalg_backend::BLAS_LAPACKE>;
template struct square_root<double, linalg_backend::BLAS_LAPACKE>;
#endif
#if LINALG_HAS_MKL
template struct square_root<float, linalg_backend::MKL>;
template struct square_root<double, linalg_backend::MKL>;
#endif
}  // namespace linalg
