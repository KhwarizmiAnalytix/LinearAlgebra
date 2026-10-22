#if LINALG_HAS_ROCBLAS

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/logarithm.h"
#include "include/matrix_operation/square_root.h"
#include "src/common/matrix_functions_workspace.h"

#define MF_INLINE __device__ inline
#define MF_ADDRESS
#include "src/common/matrix_functions_core.h"
#undef MF_ADDRESS
#undef MF_INLINE

namespace linalg
{
namespace
{
// A portable device implementation of the Schur iteration. This deliberately
// uses one thread per matrix; it prioritizes correctness and residency over
// throughput for large matrices. No solver API supplies logm/sqrtm directly.
template <typename T>
__global__ void matrix_function_kernel(
    const T* a, int n, int lda, T* result, int ldc, T* workspace, bool is_log, T epsilon)
{
    if (!linalg_matrix_functions::compute(a, n, lda, result, ldc, workspace, is_log, epsilon))
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result[i * ldc + j] = T(NAN);
            }
        }
    }
}

template <typename T>
void launch_matrix_function(const T* a,
    linalg_int                       n,
    linalg_int                       lda,
    T*                               result,
    linalg_int                       ldc,
    void*                            workspace,
    size_t                           workspace_size,
    void*                            stream,
    bool                             is_log)
{
    detail::check_matrix_function(a, n, lda, result, ldc, workspace, workspace_size);
    matrix_function_kernel<T><<<1, 1, 0, static_cast<hipStream_t>(stream)>>>(a,
        static_cast<int>(n),
        static_cast<int>(lda),
        result,
        static_cast<int>(ldc),
        static_cast<T*>(workspace),
        is_log,
        std::numeric_limits<T>::epsilon());
    LINALG_CHECK(hipGetLastError() == hipSuccess, "matrix function: HIP kernel launch failed");
}
}  // namespace

template <>
void logarithm<float, linalg_backend::ROCM_BLAS>::apply(const float* a,
    linalg_int                                                       n,
    linalg_int                                                       lda,
    float*                                                           result,
    linalg_int                                                       ldc,
    void*                                                            workspace,
    size_t                                                           workspace_size,
    void*                                                            stream)
{
    LINALG_FUNCTION();
    launch_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, true);
}

template <> size_t logarithm<float, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<float>(n);
}

template <>
void logarithm<double, linalg_backend::ROCM_BLAS>::apply(const double* a,
    linalg_int                                                         n,
    linalg_int                                                         lda,
    double*                                                            result,
    linalg_int                                                         ldc,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void*                                                              stream)
{
    LINALG_FUNCTION();
    launch_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, true);
}

template <> size_t logarithm<double, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<double>(n);
}

template <>
void square_root<float, linalg_backend::ROCM_BLAS>::apply(const float* a,
    linalg_int                                                         n,
    linalg_int                                                         lda,
    float*                                                             result,
    linalg_int                                                         ldc,
    void*                                                              workspace,
    size_t                                                             workspace_size,
    void*                                                              stream)
{
    LINALG_FUNCTION();
    launch_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, false);
}

template <> size_t square_root<float, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<float>(n);
}

template <>
void square_root<double, linalg_backend::ROCM_BLAS>::apply(const double* a,
    linalg_int                                                           n,
    linalg_int                                                           lda,
    double*                                                              result,
    linalg_int                                                           ldc,
    void*                                                                workspace,
    size_t                                                               workspace_size,
    void*                                                                stream)
{
    LINALG_FUNCTION();
    launch_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, false);
}

template <> size_t square_root<double, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<double>(n);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
