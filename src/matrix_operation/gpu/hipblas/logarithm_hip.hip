#if LINALG_HAS_ROCBLAS

#include <cmath>
#include <hip/hip_runtime.h>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/logarithm.h"
#include "src/common/logarithm_core.h"
#include "src/matrix_operation/logarithm_workspace.h"

namespace linalg
{
namespace
{
template <typename T>
__global__ void prepare_logarithm_kernel(const T* sr,
    const T*                                       si,
    int                                             n,
    T*                                              h_buffer,
    T*                                              f_buffer,
    T*                                              diagonal_buffer,
    T*                                              state,
    T                                               epsilon)
{
    linalg_schur::logarithm_schur_prepare(sr, si, n, h_buffer, f_buffer, diagonal_buffer, state, epsilon);
}

template <typename T>
__global__ void pack_logarithm_solve_kernel(
    const T* h_buffer, const T* state, T* a, T* b, int n, int node)
{
    linalg_schur::pack_logarithm_solve(h_buffer, state, a, b, n, node);
}

template <typename T>
__global__ void accumulate_logarithm_solve_kernel(
    T* f_buffer, const T* state, const T* solution, int n, int node)
{
    linalg_schur::accumulate_logarithm_solve(f_buffer, state, solution, n, node);
}

template <typename T>
__global__ void finish_logarithm_kernel(
    T* f_buffer, const T* state, const T* diagonal_buffer, int n, T* fr, T* fi)
{
    linalg_schur::logarithm_schur_finish(f_buffer, state, diagonal_buffer, n, fr, fi);
}

template <typename T>
__global__ void combine_logarithm_kernel(const T* a, const T* b, T* out, int count, T sign)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(count))
    {
        out[i] = a[i] + sign * b[i];
    }
}

template <typename T>
__global__ void publish_logarithm_kernel(
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

inline void check_logarithm_launch()
{
    LINALG_CHECK(hipGetLastError() == hipSuccess, "logarithm: kernel launch failed");
}

template <typename T, linalg_backend Backend>
void logarithm_impl(const T* a,
    linalg_int               n,
    linalg_int               lda,
    T*                       result,
    linalg_int               ldc,
    void*                    workspace,
    size_t                   workspace_size,
    void*                    stream)
{
    detail::check_logarithm_order(n);
    detail::check_schur_function_arguments<T, Backend>(a, n, lda, result, ldc, workspace,
        workspace_size, detail::logarithm_workspace_size<T, Backend>(n));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p             = detail::logarithm_plan<T, Backend>(arena, n);
    const auto              device_stream = static_cast<hipStream_t>(stream);
    const T                 epsilon       = std::numeric_limits<T>::epsilon();

    schur_decomposition<T, Backend>::apply(a, n, lda, p.schur.sr, p.schur.si, n, p.schur.qr,
        p.schur.qi, n, p.schur.schur_workspace, p.schur.schur_workspace_bytes, stream);

    prepare_logarithm_kernel<T><<<1, 1, 0, device_stream>>>(
        p.schur.sr, p.schur.si, static_cast<int>(n), p.h, p.f, p.diagonal, p.state, epsilon);
    check_logarithm_launch();
    for (int node = 0; node < 7; ++node)
    {
        pack_logarithm_solve_kernel<T><<<1, 1, 0, device_stream>>>(
            p.h, p.state, p.triangular_a, p.triangular_b, static_cast<int>(n), node);
        check_logarithm_launch();
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
            p.triangular_bytes,
            stream);
        accumulate_logarithm_solve_kernel<T><<<1, 1, 0, device_stream>>>(
            p.f, p.state, p.triangular_x, static_cast<int>(n), node);
        check_logarithm_launch();
    }
    finish_logarithm_kernel<T><<<1, 1, 0, device_stream>>>(
        p.f, p.state, p.diagonal, static_cast<int>(n), p.schur.fr, p.schur.fi);
    check_logarithm_launch();
    detail::reconstruct_schur_function<T, Backend>(p.schur,
        n,
        stream,
        [n, device_stream](const T* x, const T* y, T* out, T sign)
        {
            const auto count  = static_cast<int>(n * n);
            const auto blocks = (static_cast<unsigned>(count) + 255) / 256;
            combine_logarithm_kernel<T><<<blocks, 256, 0, device_stream>>>(x, y, out, count, sign);
            check_logarithm_launch();
        });
    publish_logarithm_kernel<T><<<1, 1, 0, device_stream>>>(
        p.schur.fr, p.schur.fi, static_cast<int>(n), result, static_cast<int>(ldc), epsilon);
    check_logarithm_launch();
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
    logarithm_impl<float, linalg_backend::ROCM_BLAS>(
        a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t logarithm<float, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    detail::check_logarithm_order(n);
    return detail::logarithm_workspace_size<float, linalg_backend::ROCM_BLAS>(n);
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
    logarithm_impl<double, linalg_backend::ROCM_BLAS>(
        a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t logarithm<double, linalg_backend::ROCM_BLAS>::workspace_size(linalg_int n)
{
    detail::check_logarithm_order(n);
    return detail::logarithm_workspace_size<double, linalg_backend::ROCM_BLAS>(n);
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
