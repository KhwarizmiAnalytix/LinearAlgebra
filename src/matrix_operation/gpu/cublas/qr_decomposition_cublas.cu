#if LINALG_HAS_CUBLAS
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"
#include "include/matrix_operation/qr_decomposition.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

namespace linalg
{
namespace
{
template <typename T>
__global__ void qr_pack(const T* A, size_t lda, T* packed, int rows, int columns)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(rows) * columns)
    {
        const auto r = i % rows;
        const auto c = i / rows;
        packed[i]    = A[r * lda + c];
    }
}
template <typename T>
__global__ void qr_unpack(const T* packed, T* A, size_t lda, int rows, int columns)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(rows) * columns)
    {
        const auto r   = i % rows;
        const auto c   = i / rows;
        A[r * lda + c] = packed[i];
    }
}

unsigned qr_blocks(int rows, int columns)
{
    return detail::checked_narrow<unsigned>((static_cast<size_t>(rows) * columns + 255) / 256);
}
template <typename T, typename Fn> int qr_r_lwork(Fn fn, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows > 0 && columns > 0, "QR: dimensions must be positive");
    const auto r     = detail::checked_narrow<int>(rows);
    const auto c     = detail::checked_narrow<int>(columns);
    int        lwork = 0;
    LINALG_CHECK(fn(detail::cusolver_handle_for_current_device(),
                     r,
                     c,
                     static_cast<T*>(nullptr),
                     r,
                     &lwork) == CUSOLVER_STATUS_SUCCESS,
        "QR geqrf workspace query failed");
    return lwork;
}
template <typename T, typename Fn> int qr_q_lwork(Fn fn, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows > 0 && columns > 0, "QR: dimensions must be positive");
    const auto r     = detail::checked_narrow<int>(rows);
    const auto k     = detail::checked_narrow<int>(std::min(rows, columns));
    int        lwork = 0;
    LINALG_CHECK(fn(detail::cusolver_handle_for_current_device(),
                     r,
                     k,
                     k,
                     static_cast<T*>(nullptr),
                     r,
                     static_cast<T*>(nullptr),
                     &lwork) == CUSOLVER_STATUS_SUCCESS,
        "QR orgqr workspace query failed");
    return lwork;
}
template <typename T> struct qr_buffers
{
    int* info;
    T*   packed;
    T*   work;
};
template <typename T>
qr_buffers<T> qr_plan(
    detail::workspace_arena& arena, linalg_long rows, linalg_long packed_columns, int lwork)
{
    qr_buffers<T> p;
    p.info   = arena.next<int>(1);
    p.packed = arena.next<T>(static_cast<size_t>(rows) * packed_columns);
    p.work   = arena.next<T>(static_cast<size_t>(lwork));
    return p;
}
template <typename T> size_t qr_size(linalg_long rows, linalg_long packed_columns, int lwork)
{
    detail::workspace_arena arena;
    qr_plan<T>(arena, rows, packed_columns, lwork);
    return arena.total_size();
}
template <typename T, typename Fn>
void qr_r(Fn    geqrf,
    linalg_long rows,
    linalg_long columns,
    T*          A,
    linalg_long lda,
    T*          tau,
    void*       workspace,
    size_t      workspace_size,
    void*       stream_ptr,
    int         lwork)
{
    LINALG_CHECK(A && tau && workspace, "QR apply_r: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_r: invalid row stride");
    LINALG_CHECK(
        workspace_size >= qr_size<T>(rows, columns, lwork), "QR apply_r: workspace too small");
    const auto r      = detail::checked_narrow<int>(rows);
    const auto c      = detail::checked_narrow<int>(columns);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p = qr_plan<T>(arena, rows, columns, lwork);
    qr_pack<<<qr_blocks(r, c), 256, 0, stream>>>(A, static_cast<size_t>(lda), p.packed, r, c);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "QR pack kernel failed");
    LINALG_CHECK(
        geqrf(handle, r, c, p.packed, r, tau, p.work, lwork, p.info) == CUSOLVER_STATUS_SUCCESS,
        "QR geqrf failed");
    qr_unpack<<<qr_blocks(r, c), 256, 0, stream>>>(p.packed, A, static_cast<size_t>(lda), r, c);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "QR unpack kernel failed");
}
template <typename T, typename Fn>
void qr_q(Fn    orgqr,
    linalg_long rows,
    linalg_long columns,
    T*          A,
    linalg_long lda,
    const T*    tau,
    void*       workspace,
    size_t      workspace_size,
    void*       stream_ptr,
    int         lwork)
{
    const auto k = detail::checked_narrow<int>(std::min(rows, columns));
    const auto r = detail::checked_narrow<int>(rows);
    LINALG_CHECK(A && tau && workspace, "QR apply_q: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_q: invalid row stride");
    LINALG_CHECK(workspace_size >= qr_size<T>(rows, k, lwork), "QR apply_q: workspace too small");
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p = qr_plan<T>(arena, rows, k, lwork);
    qr_pack<<<qr_blocks(r, k), 256, 0, stream>>>(A, static_cast<size_t>(lda), p.packed, r, k);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "QR pack kernel failed");
    LINALG_CHECK(
        orgqr(handle, r, k, k, p.packed, r, tau, p.work, lwork, p.info) == CUSOLVER_STATUS_SUCCESS,
        "QR orgqr failed");
    qr_unpack<<<qr_blocks(r, k), 256, 0, stream>>>(p.packed, A, static_cast<size_t>(lda), r, k);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess, "QR unpack kernel failed");
}
}  // namespace
template <>
size_t qr_decomposition<float, linalg_backend::cuda_blas>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_r_lwork<float>(cusolverDnSgeqrf_bufferSize, rows, columns);
    return qr_size<float>(rows, columns, lwork);
}
template <>
void qr_decomposition<float, linalg_backend::cuda_blas>::apply_r(linalg_long rows,
    linalg_long                                                              columns,
    float*                                                                   A,
    linalg_long                                                              lda,
    float*                                                                   tau,
    void*                                                                    workspace,
    size_t                                                                   workspace_size,
    void*                                                                    stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<float>/apply_r");
    const auto lwork = qr_r_lwork<float>(cusolverDnSgeqrf_bufferSize, rows, columns);
    qr_r(cusolverDnSgeqrf, rows, columns, A, lda, tau, workspace, workspace_size, stream, lwork);
}
template <>
size_t qr_decomposition<float, linalg_backend::cuda_blas>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_q_lwork<float>(cusolverDnSorgqr_bufferSize, rows, columns);
    return qr_size<float>(rows, std::min(rows, columns), lwork);
}
template <>
void qr_decomposition<float, linalg_backend::cuda_blas>::apply_q(linalg_long rows,
    linalg_long                                                              columns,
    float*                                                                   A,
    linalg_long                                                              lda,
    const float*                                                             tau,
    void*                                                                    workspace,
    size_t                                                                   workspace_size,
    void*                                                                    stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<float>/apply_q");
    const auto lwork = qr_q_lwork<float>(cusolverDnSorgqr_bufferSize, rows, columns);
    qr_q(cusolverDnSorgqr, rows, columns, A, lda, tau, workspace, workspace_size, stream, lwork);
}
template <>
size_t qr_decomposition<double, linalg_backend::cuda_blas>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_r_lwork<double>(cusolverDnDgeqrf_bufferSize, rows, columns);
    return qr_size<double>(rows, columns, lwork);
}
template <>
void qr_decomposition<double, linalg_backend::cuda_blas>::apply_r(linalg_long rows,
    linalg_long                                                               columns,
    double*                                                                   A,
    linalg_long                                                               lda,
    double*                                                                   tau,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    void*                                                                     stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<double>/apply_r");
    const auto lwork = qr_r_lwork<double>(cusolverDnDgeqrf_bufferSize, rows, columns);
    qr_r(cusolverDnDgeqrf, rows, columns, A, lda, tau, workspace, workspace_size, stream, lwork);
}
template <>
size_t qr_decomposition<double, linalg_backend::cuda_blas>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    const auto lwork = qr_q_lwork<double>(cusolverDnDorgqr_bufferSize, rows, columns);
    return qr_size<double>(rows, std::min(rows, columns), lwork);
}
template <>
void qr_decomposition<double, linalg_backend::cuda_blas>::apply_q(linalg_long rows,
    linalg_long                                                               columns,
    double*                                                                   A,
    linalg_long                                                               lda,
    const double*                                                             tau,
    void*                                                                     workspace,
    size_t                                                                    workspace_size,
    void*                                                                     stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<double>/apply_q");
    const auto lwork = qr_q_lwork<double>(cusolverDnDorgqr_bufferSize, rows, columns);
    qr_q(cusolverDnDorgqr, rows, columns, A, lda, tau, workspace, workspace_size, stream, lwork);
}
}  // namespace linalg
#endif
