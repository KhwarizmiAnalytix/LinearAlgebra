#include "include/matrix_operation/schur_decomposition.h"

#include <cmath>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "src/matrix_operation/schur_decomposition_workspace.h"

using std::isfinite;
#include "src/matrix_operation/schur_decomposition_impl.h"

namespace linalg
{
template <typename T, linalg_backend Backend>
void schur_decomposition<T, Backend>::apply(const T* a,
    linalg_int                                       n,
    linalg_int                                       lda,
    T*                                               sr,
    T*                                               si,
    linalg_int                                       lds,
    T*                                               qr,
    T*                                               qi,
    linalg_int                                       ldq,
    void*                                            workspace,
    size_t                                           bytes,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    detail::check_schur_decomposition<T, Backend>(
        a, n, lda, sr, si, lds, qr, qi, ldq, workspace, bytes);
    const bool success = linalg_schur::decompose_real(a,
        static_cast<int>(n),
        static_cast<int>(lda),
        sr,
        si,
        static_cast<int>(lds),
        qr,
        qi,
        static_cast<int>(ldq),
        static_cast<T*>(workspace),
        std::numeric_limits<T>::epsilon());
    LINALG_CHECK(success, "schur_decomposition: nonfinite input or numerical failure");
}

template <typename T, linalg_backend Backend>
size_t schur_decomposition<T, Backend>::workspace_size(linalg_int n)
{
    return detail::portable_schur_workspace_size<T>(n);
}

template struct schur_decomposition<float, linalg_backend::NATIVE>;
template struct schur_decomposition<double, linalg_backend::NATIVE>;
}  // namespace linalg
