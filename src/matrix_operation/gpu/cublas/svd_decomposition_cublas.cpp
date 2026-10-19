#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/svd_decomposition.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{

// Forward declarations for detail::svd_on_device and
// detail::svd_on_device_workspace_size, which are defined later in this file
// in namespace linalg::detail. They must be visible to the anonymous-namespace
// helpers (svd_apply) that call them.
namespace detail
{
size_t svd_on_device_workspace_size(const float*, linalg_long rows, linalg_long columns);
size_t svd_on_device_workspace_size(const double*, linalg_long rows, linalg_long columns);
void svd_on_device(linalg_long rows, linalg_long columns, const float* A, linalg_long lda,
    float* S, float* U, linalg_long ldu, float* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, cudaStream_t stream);
void svd_on_device(linalg_long rows, linalg_long columns, const double* A, linalg_long lda,
    double* S, double* U, linalg_long ldu, double* VT, linalg_long ldv,
    int* info, void* workspace, size_t workspace_size, cudaStream_t stream);
}  // namespace detail

namespace
{

// svd_core's own five scratch buffers (sized for whatever (r, c) it is
// actually invoked with — see svd_plan below for how the two possible
// invocations, direct or rows<columns-swapped, are both accounted for).
template <typename T>
struct svd_core_layout
{
    T* a_work;
    T* u_work;
    T* vt_work;
    T* work;
    T* rwork;
};

template <typename T>
svd_core_layout<T> svd_core_plan(detail::workspace_arena& arena, int r, int c, int k, int lwork)
{
    svd_core_layout<T> p;
    const auto         rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto         rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto         kc = static_cast<size_t>(k) * static_cast<size_t>(c);
    p.a_work  = arena.next<T>(rc);
    p.u_work  = arena.next<T>(rk);
    p.vt_work = arena.next<T>(kc);
    p.work    = arena.next<T>(static_cast<size_t>(lwork));
    p.rwork   = arena.next<T>(static_cast<size_t>(std::max(1, k - 1)));
    return p;
}

// Everything svd_impl needs: svd_core's own buffers (above), one shared
// transpose scratch buffer, and — only on the rows < columns path — the
// extra at/u_prime/vt_prime buffers.
template <typename T>
struct svd_layout
{
    svd_core_layout<T> core;
    T*                 transpose_scratch;
    T*                 at;        // rows < columns only
    T*                 u_prime;   // rows < columns only
    T*                 vt_prime;  // rows < columns only
};

template <typename T>
svd_layout<T> svd_plan(detail::workspace_arena& arena, int rows, int columns, int lwork)
{
    svd_layout<T> p{};
    if (rows >= columns)
    {
        p.core = svd_core_plan<T>(arena, rows, columns, columns, lwork);
    }
    else
    {
        p.core     = svd_core_plan<T>(arena, columns, rows, rows, lwork);
        const auto rc = static_cast<size_t>(rows) * static_cast<size_t>(columns);
        p.at           = arena.next<T>(rc);
        p.u_prime      = arena.next<T>(static_cast<size_t>(columns) * static_cast<size_t>(rows));
        p.vt_prime     = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(rows));
    }
    p.transpose_scratch =
        arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return p;
}

template <typename T, typename BufferSizeFn>
size_t svd_workspace_size_impl(BufferSizeFn buffer_size, linalg_long rows, linalg_long columns)
{
    LINALG_CHECK(rows != 0 && columns != 0,
        "svd_decomposition_workspace_size: rows and columns must be positive");
    const auto r      = detail::checked_narrow<int>(rows);
    const auto c      = detail::checked_narrow<int>(columns);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto eff_r  = (r >= c) ? r : c;
    const auto eff_c  = (r >= c) ? c : r;
    int        lwork  = 0;
    LINALG_CHECK(buffer_size(handle, eff_r, eff_c, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*gesvd_bufferSize failed");
    detail::workspace_arena arena;
    svd_plan<T>(arena, r, c, lwork);
    return arena.total_size();
}

// Economy SVD (jobu = jobvt = 'S') of an m x n matrix with m >= n required.
// Row-major buffers; detail::transpose_device handles the column-/row-major
// reinterpretation — same identity used throughout lu_decomposition, qr, etc.
template <typename T, typename GesvdFn>
void svd_core(GesvdFn gesvd,
    linalg_long       rows,
    linalg_long       columns,
    const T*          A,
    T*                S,
    T*                U,
    T*                VT,
    int*              info,
    const svd_core_layout<T>& bufs,
    T*                        transpose_scratch,
    int                       lwork,
    cudaStream_t              stream)
{
    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    const auto k = std::min(r, c);

    auto handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto kc = static_cast<size_t>(k) * static_cast<size_t>(c);

    LINALG_CHECK(
        cudaMemcpyAsync(bufs.a_work, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(rows, columns, bufs.a_work, transpose_scratch, stream);

    const auto status = gesvd(handle, 'S', 'S', r, c, bufs.a_work, r, S, bufs.u_work, r,
        bufs.vt_work, k, bufs.work, lwork, bufs.rwork, info);
    LINALG_CHECK(status == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*gesvd failed", static_cast<int>(status));

    LINALG_CHECK(cudaMemcpyAsync(U, bufs.u_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(static_cast<linalg_long>(k), rows, U, transpose_scratch, stream);

    LINALG_CHECK(cudaMemcpyAsync(VT, bufs.vt_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(columns, static_cast<linalg_long>(k), VT, transpose_scratch, stream);
}

template <typename T, typename BufferSizeFn, typename GesvdFn>
void svd_impl(BufferSizeFn buffer_size,
    GesvdFn      gesvd,
    linalg_long  rows,
    linalg_long  columns,
    const T*     A,
    linalg_long  lda,
    T*           S,
    T*           U,
    linalg_long  ldu,
    T*           VT,
    linalg_long  ldv,
    int*         info,
    void*        workspace,
    size_t       workspace_size,
    cudaStream_t stream)
{
    LINALG_CHECK(A != nullptr && S != nullptr && U != nullptr && VT != nullptr && info != nullptr,
        "svd_decomposition: A, S, U, VT, and info must not be null");
    LINALG_CHECK(rows != 0 && columns != 0,
        "svd_decomposition: rows and columns must be positive");

    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    const auto k = std::min(r, c);

    LINALG_CHECK(!(detail::checked_narrow<int>(lda) != c) &&
                     !(detail::checked_narrow<int>(ldu) != k) &&
                     !(detail::checked_narrow<int>(ldv) != c),
        "svd_decomposition requires tightly packed lda/ldu/ldv "
        "(lda == columns, ldu == min(rows, columns), ldv == columns)");

    auto       handle = detail::cusolver_handle_for_current_device();
    const auto eff_r   = (r >= c) ? r : c;
    const auto eff_c   = (r >= c) ? c : r;
    int        lwork   = 0;
    LINALG_CHECK(buffer_size(handle, eff_r, eff_c, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*gesvd_bufferSize failed");

    LINALG_CHECK(workspace_size >= svd_workspace_size_impl<T>(buffer_size, rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = svd_plan<T>(arena, r, c, lwork);

    if (r >= c)
    {
        svd_core(gesvd, rows, columns, A, S, U, VT, info, bufs.core, bufs.transpose_scratch, lwork,
            stream);
        return;
    }

    // rows < columns: factor A^T (columns x rows, m' >= n') instead.
    // A = (A^T)^T = (U' S' VT')^T => U = VT'^T, VT = U'^T, S = S'.
    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    LINALG_CHECK(
        cudaMemcpyAsync(bufs.at, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(rows, columns, bufs.at, bufs.transpose_scratch, stream);

    const auto ck = static_cast<size_t>(c) * static_cast<size_t>(k);
    const auto kr = static_cast<size_t>(k) * static_cast<size_t>(r);

    svd_core(gesvd, columns, rows, bufs.at, S, bufs.u_prime, bufs.vt_prime, info, bufs.core,
        bufs.transpose_scratch, lwork, stream);

    LINALG_CHECK(
        cudaMemcpyAsync(VT, bufs.u_prime, sizeof(T) * ck, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(columns, static_cast<linalg_long>(k), VT, bufs.transpose_scratch, stream);

    LINALG_CHECK(
        cudaMemcpyAsync(U, bufs.vt_prime, sizeof(T) * kr, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(static_cast<linalg_long>(k), rows, U, bufs.transpose_scratch, stream);
}

// info (1 int) is carved from the caller's device workspace ahead of the GPU
// scratch; the caller is responsible for synchronizing the stream and
// inspecting it if needed.
template <typename T>
void svd_apply(linalg_long rows,
    linalg_long  columns,
    T*           A,
    linalg_long  lda,
    T*           S,
    T*           U,
    linalg_long  ldu,
    T*           VT,
    linalg_long  ldv,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_CHECK(lda == columns && ldu == std::min(rows, columns) && ldv == columns,
        "svd_decomposition<cuda_blas>: lda/ldu/ldv must be tightly packed "
        "(lda==columns, ldu==min(rows,columns), ldv==columns)");
    const auto base =
        detail::svd_on_device_workspace_size(static_cast<const T*>(nullptr), rows, columns);
    detail::workspace_arena arena(workspace, workspace_size);
    int*  dev_info = arena.next<int>(1, alignof(int));
    void* gpu_ws   = arena.next<unsigned char>(base);
    detail::svd_on_device(rows, columns, A, lda, S, U, ldu, VT, ldv, dev_info, gpu_ws, base,
        static_cast<cudaStream_t>(stream));
}

}  // namespace

// detail::svd_on_device and detail::svd_on_device_workspace_size are
// cross-TU helpers called by norm_cublas.cu, pseudo_inverse_cublas.cu and
// rank_cublas.cpp. They live in namespace linalg::detail with external
// linkage; each calling TU forward-declares them.
namespace detail
{

size_t svd_on_device_workspace_size(const float*, linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<float>(cusolverDnSgesvd_bufferSize, rows, columns);
}

size_t svd_on_device_workspace_size(const double*, linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<double>(cusolverDnDgesvd_bufferSize, rows, columns);
}

void svd_on_device(linalg_long rows,
    linalg_long              columns,
    const float*               A,
    linalg_long              lda,
    float*                     S,
    float*                     U,
    linalg_long              ldu,
    float*                     VT,
    linalg_long              ldv,
    int*                       info,
    void*                      workspace,
    size_t                     workspace_size,
    cudaStream_t               stream)
{
    svd_impl(cusolverDnSgesvd_bufferSize, cusolverDnSgesvd,
        rows, columns, A, lda, S, U, ldu, VT, ldv, info, workspace, workspace_size, stream);
}

void svd_on_device(linalg_long rows,
    linalg_long              columns,
    const double*              A,
    linalg_long              lda,
    double*                    S,
    double*                    U,
    linalg_long              ldu,
    double*                    VT,
    linalg_long              ldv,
    int*                       info,
    void*                      workspace,
    size_t                     workspace_size,
    cudaStream_t               stream)
{
    svd_impl(cusolverDnDgesvd_bufferSize, cusolverDnDgesvd,
        rows, columns, A, lda, S, U, ldu, VT, ldv, info, workspace, workspace_size, stream);
}

}  // namespace detail

template <>
void svd_decomposition<float, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long  columns,
    float*       A,
    linalg_long  lda,
    float*       S,
    float*       U,
    linalg_long  ldu,
    float*       VT,
    linalg_long  ldv,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("svd_decomposition/cuda_blas<float>");
    svd_apply(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size, stream);
}

template <>
void svd_decomposition<double, linalg_backend::cuda_blas>::apply(linalg_long rows,
    linalg_long  columns,
    double*      A,
    linalg_long  lda,
    double*      S,
    double*      U,
    linalg_long  ldu,
    double*      VT,
    linalg_long  ldv,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("svd_decomposition/cuda_blas<double>");
    svd_apply(rows, columns, A, lda, S, U, ldu, VT, ldv, workspace, workspace_size, stream);
}

template <>
size_t svd_decomposition<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    const auto base = detail::svd_on_device_workspace_size(static_cast<const float*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    arena.next<int>(1, alignof(int));
    arena.next<unsigned char>(base);
    return arena.total_size();
}

template <>
size_t svd_decomposition<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    const auto base = detail::svd_on_device_workspace_size(static_cast<const double*>(nullptr), rows, columns);
    detail::workspace_arena arena;
    arena.next<int>(1, alignof(int));
    arena.next<unsigned char>(base);
    return arena.total_size();
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
