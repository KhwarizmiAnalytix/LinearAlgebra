#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/qr_decomposition.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

// R's strictly-below-diagonal entries are Householder-vector scratch left
// over from geqrf, not zero — this clears them, matching the CPU
// qr_decomposition's explicit "(c >= r) ? value : 0" contract. `R` is
// row-major k x columns, tightly packed (ld = columns).
template <typename scalar_t>
static __global__ void zero_qr_lower_triangle_kernel(scalar_t* R, int k, int columns)
{
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= k || col >= columns || col >= row)
    {
        return;
    }
    R[row * columns + col] = scalar_t(0);
}

namespace linalg
{
namespace
{

dim3 grid_2d(int width, int height, dim3 block)
{
    return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
        (static_cast<unsigned>(height) + block.y - 1) / block.y);
}

template <typename T>
struct qr_layout
{
    T* a_work;
    T* tau;
    T* work_geqrf;
    T* r_work;
    T* work_orgqr;
    T* q_work;
    T* transpose_scratch;
};

template <typename T>
qr_layout<T> qr_plan(
    detail::workspace_arena& arena, int r, int c, int k, int lwork_geqrf, int lwork_orgqr)
{
    qr_layout<T> p;
    p.a_work            = arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(c));
    p.tau               = arena.next<T>(static_cast<size_t>(k));
    p.work_geqrf        = arena.next<T>(static_cast<size_t>(lwork_geqrf));
    p.r_work            = arena.next<T>(static_cast<size_t>(k) * static_cast<size_t>(c));
    p.work_orgqr        = arena.next<T>(static_cast<size_t>(lwork_orgqr));
    p.q_work            = arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(k));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(r) * static_cast<size_t>(c));
    return p;
}

template <typename T, typename GeqrfBufFn, typename OrgqrBufFn>
size_t qr_workspace_size_impl(
    GeqrfBufFn geqrf_buf, OrgqrBufFn orgqr_buf, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows != 0 && columns != 0,
        "qr_decomposition_workspace_size: rows and columns must be positive");
    const auto r      = detail::checked_narrow<int>(rows);
    const auto c      = detail::checked_narrow<int>(columns);
    const auto k      = std::min(r, c);
    auto       handle = detail::cusolver_handle_for_current_device();

    int lwork_geqrf = 0;
    LINALG_CHECK(
        geqrf_buf(handle, r, c, static_cast<T*>(nullptr), r, &lwork_geqrf) ==
            CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*geqrf_bufferSize failed");
    int lwork_orgqr = 0;
    LINALG_CHECK(
        orgqr_buf(handle, r, k, k, static_cast<T*>(nullptr), r, static_cast<T*>(nullptr),
            &lwork_orgqr) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*orgqr_bufferSize failed");

    detail::workspace_arena arena;
    qr_plan<T>(arena, r, c, k, lwork_geqrf, lwork_orgqr);
    return arena.total_size();
}

template <typename T, typename GeqrfBufFn, typename GeqrfFn, typename OrgqrBufFn,
    typename OrgqrFn>
void qr_impl(GeqrfBufFn geqrf_buf,
    GeqrfFn              geqrf,
    OrgqrBufFn           orgqr_buf,
    OrgqrFn              orgqr,
    linalg_long          rows,
    linalg_long          columns,
    const T*             A,
    linalg_long          lda,
    T*                   Q,
    linalg_long          ldq,
    T*                   R,
    linalg_long          ldr,
    int*                 info,
    void*                workspace,
    size_t               workspace_size,
    cudaStream_t         stream)
{
    if (A == nullptr || Q == nullptr || R == nullptr || info == nullptr)
    {
        LINALG_THROW("qr_decomposition: A, Q, R, and info must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LINALG_THROW("qr_decomposition: rows and columns must be positive");
    }

    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    const auto k = std::min(r, c);

    if (detail::checked_narrow<int>(lda) != c || detail::checked_narrow<int>(ldq) != k ||
        detail::checked_narrow<int>(ldr) != c)
    {
        LINALG_THROW("qr_decomposition<cuda_blas> requires tightly packed lda/ldq/ldr "
                     "(lda == columns, ldq == min(rows, columns), ldr == columns)");
    }

    auto handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto kc = static_cast<size_t>(k) * static_cast<size_t>(c);

    int lwork_geqrf = 0;
    LINALG_CHECK(
        geqrf_buf(handle, r, c, static_cast<T*>(nullptr), r, &lwork_geqrf) ==
            CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*geqrf_bufferSize failed");
    int lwork_orgqr = 0;
    LINALG_CHECK(
        orgqr_buf(handle, r, k, k, static_cast<const T*>(nullptr), r,
            static_cast<const T*>(nullptr), &lwork_orgqr) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*orgqr_bufferSize failed");

    LINALG_CHECK(workspace_size >= qr_workspace_size_impl<T>(geqrf_buf, orgqr_buf, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = qr_plan<T>(arena, r, c, k, lwork_geqrf, lwork_orgqr);

    LINALG_CHECK(
        cudaMemcpyAsync(bufs.a_work, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(rows, columns, bufs.a_work, bufs.transpose_scratch, stream);

    const auto geqrf_status =
        geqrf(handle, r, c, bufs.a_work, r, bufs.tau, bufs.work_geqrf, lwork_geqrf, info);
    LINALG_CHECK(geqrf_status == CUSOLVER_STATUS_SUCCESS, "cusolverDn*geqrf failed");

    LINALG_CHECK(cudaMemcpy2DAsync(bufs.r_work,
                     sizeof(T) * static_cast<size_t>(k),
                     bufs.a_work,
                     sizeof(T) * static_cast<size_t>(r),
                     sizeof(T) * static_cast<size_t>(k),
                     static_cast<size_t>(c),
                     cudaMemcpyDeviceToDevice,
                     stream) == cudaSuccess,
        "cudaMemcpy2DAsync failed");
    detail::transpose_device(
        static_cast<linalg_long>(c), static_cast<linalg_long>(k), bufs.r_work,
        bufs.transpose_scratch, stream);

    const dim3 block(16, 16);
    zero_qr_lower_triangle_kernel<T>
        <<<grid_2d(c, k, block), block, 0, stream>>>(bufs.r_work, k, c);
    LINALG_CHECK(cudaGetLastError() == cudaSuccess,
        "zero_qr_lower_triangle_kernel launch failed");

    LINALG_CHECK(
        cudaMemcpyAsync(R, bufs.r_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");

    const auto orgqr_status =
        orgqr(handle, r, k, k, bufs.a_work, r, bufs.tau, bufs.work_orgqr, lwork_orgqr, info);
    LINALG_CHECK(orgqr_status == CUSOLVER_STATUS_SUCCESS, "cusolverDn*orgqr failed");

    LINALG_CHECK(
        cudaMemcpyAsync(bufs.q_work, bufs.a_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice,
            stream) == cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(
        static_cast<linalg_long>(k), rows, bufs.q_work, bufs.transpose_scratch, stream);

    LINALG_CHECK(
        cudaMemcpyAsync(Q, bufs.q_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
}

// The struct workspace_size() adds one int (dev_info) in front of the GPU
// workspace, matching the pattern used by lu_decomposition and
// svd_decomposition cublas adapters.
template <typename T, typename GeqrfBufFn, typename OrgqrBufFn>
size_t qr_ws_size(GeqrfBufFn geqrf_buf, OrgqrBufFn orgqr_buf,
    linalg_long rows, linalg_long columns)
{
    const auto gpu_base = qr_workspace_size_impl<T>(geqrf_buf, orgqr_buf, rows, columns);
    detail::workspace_arena arena;
    arena.next<int>(1, alignof(int));
    arena.next<unsigned char>(gpu_base);
    return arena.total_size();
}

template <typename T, typename GeqrfBufFn, typename GeqrfFn, typename OrgqrBufFn,
    typename OrgqrFn>
void qr_apply(GeqrfBufFn geqrf_buf,
    GeqrfFn              geqrf,
    OrgqrBufFn           orgqr_buf,
    OrgqrFn              orgqr,
    linalg_long          rows,
    linalg_long          columns,
    const T*             A,
    linalg_long          lda,
    T*                   Q,
    linalg_long          ldq,
    T*                   R,
    linalg_long          ldr,
    void*                workspace,
    size_t               workspace_size,
    void*                stream)
{
    const linalg_long k = rows < columns ? rows : columns;
    LINALG_CHECK(lda == columns && ldq == k && ldr == columns,
        "qr_decomposition<cuda_blas>: lda/ldq/ldr must be tightly packed "
        "(lda==columns, ldq==min(rows,columns), ldr==columns)");
    const auto gpu_base =
        qr_workspace_size_impl<T>(geqrf_buf, orgqr_buf, rows, columns);
    detail::workspace_arena arena(workspace, workspace_size);
    int*  dev_info = arena.next<int>(1, alignof(int));
    void* gpu_ws   = arena.next<unsigned char>(gpu_base);
    qr_impl(geqrf_buf, geqrf, orgqr_buf, orgqr, rows, columns, A, lda, Q, ldq, R, ldr,
        dev_info, gpu_ws, gpu_base, static_cast<cudaStream_t>(stream));
}

}  // namespace

template <>
void qr_decomposition<float, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long  columns,
    const float* A,
    linalg_long  lda,
    float*       Q,
    linalg_long  ldq,
    float*       R,
    linalg_long  ldr,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<float>");
    qr_apply(cusolverDnSgeqrf_bufferSize, cusolverDnSgeqrf, cusolverDnSorgqr_bufferSize,
        cusolverDnSorgqr, rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size,
        stream);
}

template <>
void qr_decomposition<double, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long   columns,
    const double* A,
    linalg_long   lda,
    double*       Q,
    linalg_long   ldq,
    double*       R,
    linalg_long   ldr,
    void*         workspace,
    size_t        workspace_size,
    void*         stream)
{
    LINALG_SCOPE("qr_decomposition/cuda_blas<double>");
    qr_apply(cusolverDnDgeqrf_bufferSize, cusolverDnDgeqrf, cusolverDnDorgqr_bufferSize,
        cusolverDnDorgqr, rows, columns, A, lda, Q, ldq, R, ldr, workspace, workspace_size,
        stream);
}

template <>
size_t qr_decomposition<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_ws_size<float>(cusolverDnSgeqrf_bufferSize, cusolverDnSorgqr_bufferSize,
        rows, columns);
}

template <>
size_t qr_decomposition<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return qr_ws_size<double>(cusolverDnDgeqrf_bufferSize, cusolverDnDorgqr_bufferSize,
        rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
