#include "include/matrix_operation_gpu/qr_decomposition_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"
#include "include/util/exception.h"

// R's strictly-below-diagonal entries are Householder-vector scratch left
// over from geqrf, not zero — this clears them, matching the CPU
// qr_decomposition's explicit "(c >= r) ? value : 0" contract. `R` is
// row-major k x columns, tightly packed (ld = columns), matching
// matrix_inversion_gpu.cu's convention of a single-purpose file-scope
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

// See qr_decomposition_gpu.h's RESTRICTION note: every caller-visible
// buffer here is required to be tightly packed row-major.
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

    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);
    const auto k = std::min(r, c);

    if (static_cast<int>(lda) != c || static_cast<int>(ldq) != k || static_cast<int>(ldr) != c)
    {
        LINALG_THROW("linalg::gpu::qr_decomposition requires tightly packed lda/ldq/ldr "
                     "(lda == columns, ldq == min(rows, columns), ldr == columns)");
    }

    auto handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto kc = static_cast<size_t>(k) * static_cast<size_t>(c);

    T* a_work = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&a_work), sizeof(T) * rc) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMemcpyAsync(a_work, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
    {
        cudaFree(a_work);
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    // Same identity svd_decomposition_gpu.cxx relies on: transposing the
    // row-major buffer first makes cuSOLVER's column-major (m=rows,
    // n=columns, lda=rows) reading of it equal A.
    matrix_transpose(rows, columns, a_work, stream);

    T* tau = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&tau), sizeof(T) * static_cast<size_t>(k)) != cudaSuccess)
    {
        cudaFree(a_work);
        LINALG_THROW("cudaMalloc failed");
    }

    int lwork_geqrf = 0;
    if (geqrf_buffer_size(handle, r, c, a_work, r, &lwork_geqrf) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cusolverDn*geqrf_bufferSize failed");
    }
    T* work_geqrf = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&work_geqrf),
            sizeof(T) * static_cast<size_t>(lwork_geqrf)) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cudaMalloc failed");
    }

    const auto geqrf_status = geqrf(handle, r, c, a_work, r, tau, work_geqrf, lwork_geqrf, info);
    cudaFree(work_geqrf);
    if (geqrf_status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cusolverDn*geqrf failed");
    }

    // Extract R: the top k rows of a_work (column-major, r x c, lda = r),
    // across all c columns, into a tightly-packed column-major (k x c)
    // buffer — a genuine strided 2D copy (column-major storage means each
    // source column's first k elements are contiguous with each other but
    // not with the next column's).
    T* r_work = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&r_work), sizeof(T) * kc) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMemcpy2DAsync(r_work,
            sizeof(T) * static_cast<size_t>(k),
            a_work,
            sizeof(T) * static_cast<size_t>(r),
            sizeof(T) * static_cast<size_t>(k),
            static_cast<size_t>(c),
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        cudaFree(r_work);
        LINALG_THROW("cudaMemcpy2DAsync failed");
    }
    // r_work's bytes, column-major (k x c), equal row-major (c x k) of the
    // same logical matrix transposed (the usual identity) — reinterpreting
    // costs nothing; matrix_transpose turns that into genuine row-major
    // (k x c) = R.
    matrix_transpose(static_cast<linalg_long>(c), static_cast<linalg_long>(k), r_work, stream);

    const dim3 block(16, 16);
    zero_qr_lower_triangle_kernel<T><<<grid_2d(c, k, block), block, 0, stream>>>(r_work, k, c);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        cudaFree(r_work);
        LINALG_THROW("zero_qr_lower_triangle_kernel launch failed");
    }

    const auto r_copy_status =
        cudaMemcpyAsync(R, r_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream);
    cudaFree(r_work);
    if (r_copy_status != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cudaMemcpyAsync failed");
    }

    // Extract Q: orgqr(m=r, n=k, k_reflectors=k) overwrites a_work's first
    // k columns (column-major, lda = r) with Q — those k columns are the
    // buffer's first r*k elements (column-major storage is contiguous
    // column-by-column), so no strided copy is needed here.
    int lwork_orgqr = 0;
    if (orgqr_buffer_size(handle, r, k, k, a_work, r, tau, &lwork_orgqr) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cusolverDn*orgqr_bufferSize failed");
    }
    T* work_orgqr = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&work_orgqr),
            sizeof(T) * static_cast<size_t>(lwork_orgqr)) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(tau);
        LINALG_THROW("cudaMalloc failed");
    }

    const auto orgqr_status = orgqr(handle, r, k, k, a_work, r, tau, work_orgqr, lwork_orgqr, info);
    cudaFree(tau);
    cudaFree(work_orgqr);
    if (orgqr_status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(a_work);
        LINALG_THROW("cusolverDn*orgqr failed");
    }

    T* q_work = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&q_work), sizeof(T) * rk) != cudaSuccess)
    {
        cudaFree(a_work);
        LINALG_THROW("cudaMalloc failed");
    }
    const auto q_copy_status =
        cudaMemcpyAsync(q_work, a_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream);
    cudaFree(a_work);
    if (q_copy_status != cudaSuccess)
    {
        cudaFree(q_work);
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    // Same reinterpret-then-transpose identity used for R above.
    matrix_transpose(static_cast<linalg_long>(k), rows, q_work, stream);

    const auto q_final_copy_status =
        cudaMemcpyAsync(Q, q_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream);
    cudaFree(q_work);
    if (q_final_copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync failed");
    }
}

}  // namespace

void qr_decomposition(linalg_long rows,
    linalg_long                   columns,
    const float*                    A,
    linalg_long                   lda,
    float*                          Q,
    linalg_long                   ldq,
    float*                          R,
    linalg_long                   ldr,
    int*                            info,
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
        stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
