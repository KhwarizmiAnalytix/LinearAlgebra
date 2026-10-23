#if LINALG_HAS_ROCBLAS

#include <cmath>
#include <hip/hip_runtime.h>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/square_root.h"
#include "src/common/square_root_core.h"
#include "src/matrix_operation/square_root_workspace.h"

namespace linalg
{
namespace
{
template <typename T>
__global__ void square_root_schur_kernel(
    const T* sr, const T* si, int n, T* h_buffer, T* fr, T* fi, T epsilon)
{
    linalg_schur::square_root_schur(sr, si, n, h_buffer, fr, fi, epsilon);
}

template <typename T>
__global__ void combine_square_root_kernel(const T* a, const T* b, T* out, int count, T sign)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(count))
    {
        out[i] = a[i] + sign * b[i];
    }
}

template <typename T>
__global__ void publish_square_root_kernel(
    const T* real, const T* imag, int n, T* result, int ldc, T epsilon)
{
    if (!linalg_schur::publish(real, imag, n, result, ldc, epsilon))
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

inline void check_square_root_launch()
{
    LINALG_CHECK(hipGetLastError() == hipSuccess, "square_root: kernel launch failed");
}

template <typename T, linalg_backend Backend>
void square_root_impl(const T* a,
    linalg_int                 n,
    linalg_int                 lda,
    T*                         result,
    linalg_int                 ldc,
    void*                      workspace,
    size_t                     workspace_size,
    void*                      stream)
{
    detail::check_schur_function_arguments<T, Backend>(a, n, lda, result, ldc, workspace,
        workspace_size, detail::square_root_workspace_size<T, Backend>(n));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p             = detail::square_root_plan<T, Backend>(arena, n);
    const auto              device_stream = static_cast<hipStream_t>(stream);
    const T                 epsilon       = std::numeric_limits<T>::epsilon();

    schur_decomposition<T, Backend>::apply(a, n, lda, p.schur.sr, p.schur.si, n, p.schur.qr,
        p.schur.qi, n, p.schur.schur_workspace, p.schur.schur_workspace_bytes, stream);

    square_root_schur_kernel<T><<<1, 1, 0, device_stream>>>(
        p.schur.sr, p.schur.si, static_cast<int>(n), p.h, p.schur.fr, p.schur.fi, epsilon);
    check_square_root_launch();
    detail::reconstruct_schur_function<T, Backend>(p.schur,
        n,
        stream,
        [n, device_stream](const T* x, const T* y, T* out, T sign)
        {
            const auto count  = static_cast<int>(n * n);
            const auto blocks = (static_cast<unsigned>(count) + 255) / 256;
            combine_square_root_kernel<T>
                <<<blocks, 256, 0, device_stream>>>(x, y, out, count, sign);
            check_square_root_launch();
        });
    publish_square_root_kernel<T><<<1, 1, 0, device_stream>>>(
        p.schur.fr, p.schur.fi, static_cast<int>(n), result, static_cast<int>(ldc), epsilon);
    check_square_root_launch();
}

}  // namespace

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
    square_root_impl<float, linalg_backend::ROCM_BLAS>(
        a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t square_root<float, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::square_root_workspace_size<float, linalg_backend::ROCM_BLAS>(n);
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
    square_root_impl<double, linalg_backend::ROCM_BLAS>(
        a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t square_root<double, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    return detail::square_root_workspace_size<double, linalg_backend::ROCM_BLAS>(n);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
