#include "include/matrix_operation/logarithm.h"
#include "include/matrix_operation/square_root.h"

#include <cmath>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "src/common/matrix_functions_workspace.h"

// The kernel body also compiles as CUDA/HIP device code and Metal source.
using std::isfinite;
#define MF_INLINE inline
#define MF_ADDRESS
#include "src/common/matrix_functions_core.h"
#undef MF_ADDRESS
#undef MF_INLINE

namespace linalg
{
namespace
{
template <typename T>
void apply_matrix_function(const T* a,
    linalg_int                      n,
    linalg_int                      lda,
    T*                              result,
    linalg_int                      ldc,
    void*                           workspace,
    size_t                          workspace_size,
    bool                            is_log)
{
    detail::check_matrix_function(a, n, lda, result, ldc, workspace, workspace_size);
    const bool success = linalg_matrix_functions::compute(a,
        static_cast<int>(n),
        static_cast<int>(lda),
        result,
        static_cast<int>(ldc),
        static_cast<T*>(workspace),
        is_log,
        std::numeric_limits<T>::epsilon());
    LINALG_CHECK(success,
        "matrix {}: nonfinite input, unsupported real domain, or numerical failure",
        is_log ? "logarithm" : "square root");
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
    apply_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, true);
}

template <typename T, linalg_backend Backend>
size_t logarithm<T, Backend>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<T>(n);
}

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
    apply_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, false);
}

template <typename T, linalg_backend Backend>
size_t square_root<T, Backend>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<T>(n);
}

#define LINALG_INSTANTIATE_MATRIX_FUNCTIONS(backend)                                               \
    template struct logarithm<float, backend>;                                                     \
    template struct logarithm<double, backend>;                                                    \
    template struct square_root<float, backend>;                                                   \
    template struct square_root<double, backend>;

LINALG_INSTANTIATE_MATRIX_FUNCTIONS(linalg_backend::NATIVE)
#if LINALG_HAS_BLAS
LINALG_INSTANTIATE_MATRIX_FUNCTIONS(linalg_backend::BLAS_LAPACKE)
#endif
#if LINALG_HAS_MKL
LINALG_INSTANTIATE_MATRIX_FUNCTIONS(linalg_backend::MKL)
#endif
#undef LINALG_INSTANTIATE_MATRIX_FUNCTIONS

}  // namespace linalg
