#include "include/matrix_operation_gpu/qr_decomposition_gpu.h"

#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

// R's strictly-below-diagonal entries are Householder-vector scratch left
// over from geqrf, not zero — this clears them, matching the CPU
// qr_decomposition's explicit "(c >= r) ? value : 0" contract. `R` is
// row-major k x columns, tightly packed (ld = columns), matching
// inversion_gpu.cu's convention of a single-purpose file-scope
// kernel per translation unit.
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
namespace gpu
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
    T* a_work;             // r*c
    T* tau;                // k
    T* work_geqrf;         // lwork_geqrf
    T* r_work;             // k*c
    T* work_orgqr;         // lwork_orgqr
    T* q_work;             // r*k
    T* transpose_scratch;  // r*c: the largest of the three transpose calls' needs
                           // (r*c, k*c, k*r), all <= r*c since k <= r and k <= c
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
    GeqrfBufFn geqrf_buffer_size, OrgqrBufFn orgqr_buffer_size, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows != 0 && columns != 0, "qr_decomposition_workspace_size: rows and columns must be positive");
    const auto r      = detail::checked_narrow<int>(rows);
    const auto c      = detail::checked_narrow<int>(columns);
    const auto k      = std::min(r, c);
    auto       handle = detail::cusolver_handle_for_current_device();

    int lwork_geqrf = 0;
    LINALG_CHECK(geqrf_buffer_size(handle, r, c, static_cast<T*>(nullptr), r, &lwork_geqrf) ==
            CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*geqrf_bufferSize failed");
    int lwork_orgqr = 0;
    LINALG_CHECK(orgqr_buffer_size(handle, r, k, k, static_cast<T*>(nullptr), r,
                      static_cast<T*>(nullptr), &lwork_orgqr) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*orgqr_bufferSize failed");

    detail::workspace_arena arena;
    qr_plan<T>(arena, r, c, k, lwork_geqrf, lwork_orgqr);
    return arena.total_size();
}

// See qr_decomposition_gpu.h's RESTRICTION note: every caller-visible
// buffer here is required to be tightly packed row-major. `workspace`/
// `workspace_size` are caller-owned; keep them alive and unmodified until
// work queued on `stream` has completed.
template <typename T, typename GeqrfBufFn, typename GeqrfFn, typename OrgqrBufFn, typename OrgqrFn>
void qr_impl(GeqrfBufFn geqrf_buffer_size,
    GeqrfFn                geqrf,
    OrgqrBufFn             orgqr_buffer_size,
    OrgqrFn                orgqr,
    linalg_long          rows,
    linalg_long          columns,
    const T*               A,
    linalg_long          lda,
    T*                     Q,
    linalg_long          ldq,
    T*                     R,
    linalg_long          ldr,
    int*                   info,
    void*                  workspace,
    size_t                 workspace_size,
    cudaStream_t           stream)
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

    if (detail::checked_narrow<int>(lda) != c || detail::checked_narrow<int>(ldq) != k || detail::checked_narrow<int>(ldr) != c)
    {
        LINALG_THROW("linalg::gpu::qr_decomposition requires tightly packed lda/ldq/ldr "
                     "(lda == columns, ldq == min(rows, columns), ldr == columns)");
    }

    auto handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto kc = static_cast<size_t>(k) * static_cast<size_t>(c);

    int lwork_geqrf = 0;
    LINALG_CHECK(geqrf_buffer_size(handle, r, c, static_cast<const T*>(nullptr), r, &lwork_geqrf) ==
            CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*geqrf_bufferSize failed");
    int lwork_orgqr = 0;
    LINALG_CHECK(orgqr_buffer_size(handle, r, k, k, static_cast<const T*>(nullptr), r,
                      static_cast<const T*>(nullptr), &lwork_orgqr) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*orgqr_bufferSize failed");

    LINALG_CHECK(workspace_size >= qr_workspace_size_impl<T>(
                                         geqrf_buffer_size, orgqr_buffer_size, rows, columns),
        "qr_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = qr_plan<T>(arena, r, c, k, lwork_geqrf, lwork_orgqr);

    LINALG_CHECK(cudaMemcpyAsync(bufs.a_work, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    // Same identity svd_decomposition_gpu.cpp relies on: transposing the
    // row-major buffer first makes cuSOLVER's column-major (m=rows,
    // n=columns, lda=rows) reading of it equal A.
    detail::transpose_device(rows, columns, bufs.a_work, bufs.transpose_scratch, stream);

    const auto geqrf_status =
        geqrf(handle, r, c, bufs.a_work, r, bufs.tau, bufs.work_geqrf, lwork_geqrf, info);
    LINALG_CHECK(geqrf_status == CUSOLVER_STATUS_SUCCESS, "cusolverDn*geqrf failed");

    // Extract R: the top k rows of a_work (column-major, r x c, lda = r),
    // across all c columns, into a tightly-packed column-major (k x c)
    // buffer — a genuine strided 2D copy (column-major storage means each
    // source column's first k elements are contiguous with each other but
    // not with the next column's).
    LINALG_CHECK(cudaMemcpy2DAsync(bufs.r_work,
                      sizeof(T) * static_cast<size_t>(k),
                      bufs.a_work,
                      sizeof(T) * static_cast<size_t>(r),
                      sizeof(T) * static_cast<size_t>(k),
                      static_cast<size_t>(c),
                      cudaMemcpyDeviceToDevice,
                      stream) == cudaSuccess,
        "cudaMemcpy2DAsync failed");
    // r_work's bytes, column-major (k x c), equal row-major (c x k) of the
    // same logical matrix transposed (the usual identity) — reinterpreting
    // costs nothing; transpose_device turns that into genuine row-major
    // (k x c) = R.
    detail::transpose_device(static_cast<linalg_long>(c), static_cast<linalg_long>(k), bufs.r_work,
        bufs.transpose_scratch, stream);

    const dim3 block(16, 16);
    zero_qr_lower_triangle_kernel<T><<<grid_2d(c, k, block), block, 0, stream>>>(bufs.r_work, k, c);
    LINALG_CHECK(
        cudaGetLastError() == cudaSuccess, "zero_qr_lower_triangle_kernel launch failed");

    LINALG_CHECK(cudaMemcpyAsync(R, bufs.r_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");

    // Extract Q: orgqr(m=r, n=k, k_reflectors=k) overwrites a_work's first
    // k columns (column-major, lda = r) with Q — those k columns are the
    // buffer's first r*k elements (column-major storage is contiguous
    // column-by-column), so no strided copy is needed here.
    const auto orgqr_status =
        orgqr(handle, r, k, k, bufs.a_work, r, bufs.tau, bufs.work_orgqr, lwork_orgqr, info);
    LINALG_CHECK(orgqr_status == CUSOLVER_STATUS_SUCCESS, "cusolverDn*orgqr failed");

    LINALG_CHECK(cudaMemcpyAsync(bufs.q_work, bufs.a_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice,
                      stream) == cudaSuccess,
        "cudaMemcpyAsync failed");
    // Same reinterpret-then-transpose identity used for R above.
    detail::transpose_device(static_cast<linalg_long>(k), rows, bufs.q_work, bufs.transpose_scratch, stream);

    LINALG_CHECK(
        cudaMemcpyAsync(Q, bufs.q_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
}

}  // namespace

size_t qr_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<float>(
        cusolverDnSgeqrf_bufferSize, cusolverDnSorgqr_bufferSize, rows, columns);
}

size_t qr_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return qr_workspace_size_impl<double>(
        cusolverDnDgeqrf_bufferSize, cusolverDnDorgqr_bufferSize, rows, columns);
}

void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const float*                    A,
    linalg_long                   lda,
    float*                          Q,
    linalg_long                   ldq,
    float*                          R,
    linalg_long                   ldr,
    int*                            info,
    void*                           workspace,
    size_t                          workspace_size,
    cudaStream_t                    stream)
{
    qr_impl(cusolverDnSgeqrf_bufferSize,
        cusolverDnSgeqrf,
        cusolverDnSorgqr_bufferSize,
        cusolverDnSorgqr,
        rows,
        columns,
        A,
        lda,
        Q,
        ldq,
        R,
        ldr,
        info,
        workspace,
        workspace_size,
        stream);
}

void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const double*                   A,
    linalg_long                   lda,
    double*                         Q,
    linalg_long                   ldq,
    double*                         R,
    linalg_long                   ldr,
    int*                            info,
    void*                           workspace,
    size_t                          workspace_size,
    cudaStream_t                    stream)
{
    qr_impl(cusolverDnDgeqrf_bufferSize,
        cusolverDnDgeqrf,
        cusolverDnDorgqr_bufferSize,
        cusolverDnDorgqr,
        rows,
        columns,
        A,
        lda,
        Q,
        ldq,
        R,
        ldr,
        info,
        workspace,
        workspace_size,
        stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
