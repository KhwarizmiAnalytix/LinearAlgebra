#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{
namespace
{

// Economy SVD (jobu = jobvt = 'S') of an m x n matrix, m >= n required —
// cuSOLVER's classic cusolverDn<t>gesvd only supports that case (unlike
// LAPACK's gesvd, which handles either orientation directly); the m < n
// case is handled one level up, in svd_impl, by factoring A^T instead. `A`
// is tightly packed row-major, read-only (copied to internal scratch
// before factoring, matching the CPU MKL/BLAS backends' contract of not
// destroying the caller's input). `S` (length k = min(m,n) = n here),
// `U` (m x k, tightly packed row-major) and `VT` (k x n, tightly packed
// row-major) are written directly — `S` needs no layout conversion (a
// plain vector), `U`/`VT` are converted from cuSOLVER's column-major
// output via matrix_transpose_gpu, the same identity used throughout
// lu_decomposition_gpu.cxx and matrix_inversion_gpu.cu: converting
// column-major storage of an (RX x CX) matrix to row-major storage of the
// same matrix is matrix_transpose_gpu(CX, RX, buf) — swapped dimensions
// relative to the matrix's own true shape.
template <typename T, typename BufferSizeFn, typename GesvdFn>
void svd_core(BufferSizeFn buffer_size,
    GesvdFn                gesvd,
    linalg_long          rows,
    linalg_long          columns,
    const T*               A,
    T*                     S,
    T*                     U,
    T*                     VT,
    int*                   info,
    cudaStream_t           stream)
{
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);
    const auto k = std::min(r, c);  // == c, since the caller guarantees rows >= columns

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
    // See lu_decomposition_gpu.cxx: transposing the row-major buffer first
    // makes cuSOLVER's column-major (m=rows, n=columns, lda=rows) reading
    // of it equal A.
    matrix_transpose(rows, columns, a_work, stream);

    T* u_work  = nullptr;
    T* vt_work = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&u_work), sizeof(T) * rk) != cudaSuccess)
    {
        cudaFree(a_work);
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&vt_work), sizeof(T) * kc) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(u_work);
        LINALG_THROW("cudaMalloc failed");
    }

    int lwork = 0;
    if (buffer_size(handle, r, c, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(a_work);
        cudaFree(u_work);
        cudaFree(vt_work);
        LINALG_THROW("cusolverDn*gesvd_bufferSize failed");
    }

    T* work = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&work), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(u_work);
        cudaFree(vt_work);
        LINALG_THROW("cudaMalloc failed");
    }
    // Real-valued gesvd's rwork (superdiagonal scratch): dimension
    // min(m,n) - 1, but never zero-sized even when k == 1.
    T* rwork = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&rwork),
            sizeof(T) * static_cast<size_t>(std::max(1, k - 1))) != cudaSuccess)
    {
        cudaFree(a_work);
        cudaFree(u_work);
        cudaFree(vt_work);
        cudaFree(work);
        LINALG_THROW("cudaMalloc failed");
    }

    const auto status = gesvd(
        handle, 'S', 'S', r, c, a_work, r, S, u_work, r, vt_work, k, work, lwork, rwork, info);

    cudaFree(a_work);
    cudaFree(work);
    cudaFree(rwork);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(u_work);
        cudaFree(vt_work);
        LINALG_THROW("cusolverDn*gesvd failed", static_cast<int>(status));
    }

    const auto u_copy_status =
        cudaMemcpyAsync(U, u_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream);
    cudaFree(u_work);
    if (u_copy_status != cudaSuccess)
    {
        cudaFree(vt_work);
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    matrix_transpose(static_cast<linalg_long>(k), rows, U, stream);

    const auto vt_copy_status =
        cudaMemcpyAsync(VT, vt_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream);
    cudaFree(vt_work);
    if (vt_copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    matrix_transpose(columns, static_cast<linalg_long>(k), VT, stream);
}

// See svd_decomposition_gpu.h's RESTRICTION note: every caller-visible
// buffer here is required to be tightly packed row-major.
template <typename T, typename BufferSizeFn, typename GesvdFn>
void svd_impl(BufferSizeFn buffer_size,
    GesvdFn                gesvd,
    linalg_long          rows,
    linalg_long          columns,
    const T*               A,
    linalg_long          lda,
    T*                     S,
    T*                     U,
    linalg_long          ldu,
    T*                     VT,
    linalg_long          ldv,
    int*                   info,
    cudaStream_t           stream)
{
    if (A == nullptr || S == nullptr || U == nullptr || VT == nullptr || info == nullptr)
    {
        LINALG_THROW("svd_decomposition: A, S, U, VT, and info must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LINALG_THROW("svd_decomposition: rows and columns must be positive");
    }

    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);
    const auto k = std::min(r, c);

    if (static_cast<int>(lda) != c || static_cast<int>(ldu) != k || static_cast<int>(ldv) != c)
    {
        LINALG_THROW("linalg::gpu::svd_decomposition requires tightly packed lda/ldu/ldv "
                     "(lda == columns, ldu == min(rows, columns), ldv == columns)");
    }

    if (r >= c)
    {
        svd_core(buffer_size, gesvd, rows, columns, A, S, U, VT, info, stream);
        return;
    }

    // cuSOLVER's classic gesvd only supports m >= n (see svd_core), so for
    // rows < columns this factors A^T (columns x rows, which does satisfy
    // m' >= n') instead and relabels: A = (A^T)^T = (U' S' VT')^T
    // = VT'^T S' U'^T, so this A's U = VT'^T (rows x k) and
    // VT = U'^T (k x columns); S = S' unchanged (SVD singular values are
    // transpose-invariant).
    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);

    T* at = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&at), sizeof(T) * rc) != cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMemcpyAsync(at, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
    {
        cudaFree(at);
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    matrix_transpose(rows, columns, at, stream);  // at becomes row-major (columns x rows) = A^T

    const auto ck = static_cast<size_t>(c) * static_cast<size_t>(k);  // columns * k (k == rows)
    const auto kr = static_cast<size_t>(k) * static_cast<size_t>(r);  // k * rows

    T* u_prime  = nullptr;  // columns x k
    T* vt_prime = nullptr;  // k x rows
    if (cudaMalloc(reinterpret_cast<void**>(&u_prime), sizeof(T) * ck) != cudaSuccess)
    {
        cudaFree(at);
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&vt_prime), sizeof(T) * kr) != cudaSuccess)
    {
        cudaFree(at);
        cudaFree(u_prime);
        LINALG_THROW("cudaMalloc failed");
    }

    svd_core(buffer_size, gesvd, columns, rows, at, S, u_prime, vt_prime, info, stream);
    cudaFree(at);

    // Caller's VT (k x columns) = transpose(u_prime (columns x k));
    // u_prime is already row-major (svd_core's own output convention), so
    // this is a genuine transpose, not the column-/row-major
    // reinterpretation trick used elsewhere in this file.
    const auto vt_copy_status =
        cudaMemcpyAsync(VT, u_prime, sizeof(T) * ck, cudaMemcpyDeviceToDevice, stream);
    cudaFree(u_prime);
    if (vt_copy_status != cudaSuccess)
    {
        cudaFree(vt_prime);
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    matrix_transpose(columns, static_cast<linalg_long>(k), VT, stream);

    // Caller's U (rows x k) = transpose(vt_prime (k x rows)).
    const auto u_copy_status =
        cudaMemcpyAsync(U, vt_prime, sizeof(T) * kr, cudaMemcpyDeviceToDevice, stream);
    cudaFree(vt_prime);
    if (u_copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync failed");
    }
    matrix_transpose(static_cast<linalg_long>(k), rows, U, stream);
}

}  // namespace

void svd_decomposition(linalg_long rows,
    linalg_long                    columns,
    const float*                     A,
    linalg_long                    lda,
    float*                           S,
    float*                           U,
    linalg_long                    ldu,
    float*                           VT,
    linalg_long                    ldv,
    int*                             info,
    cudaStream_t                     stream)
{
    svd_impl(cusolverDnSgesvd_bufferSize,
        cusolverDnSgesvd,
        rows,
        columns,
        A,
        lda,
        S,
        U,
        ldu,
        VT,
        ldv,
        info,
        stream);
}

void svd_decomposition(linalg_long rows,
    linalg_long                    columns,
    const double*                    A,
    linalg_long                    lda,
    double*                          S,
    double*                          U,
    linalg_long                    ldu,
    double*                          VT,
    linalg_long                    ldv,
    int*                             info,
    cudaStream_t                     stream)
{
    svd_impl(cusolverDnDgesvd_bufferSize,
        cusolverDnDgesvd,
        rows,
        columns,
        A,
        lda,
        S,
        U,
        ldu,
        VT,
        ldv,
        info,
        stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
