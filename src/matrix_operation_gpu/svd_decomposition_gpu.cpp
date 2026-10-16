#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"

#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include "include/logging.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace gpu
{
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
    // Real-valued gesvd's rwork (superdiagonal scratch): dimension
    // min(m,n) - 1, but never zero-sized even when k == 1.
    p.rwork = arena.next<T>(static_cast<size_t>(std::max(1, k - 1)));
    return p;
}

// Everything svd_impl needs: svd_core's own buffers (above), one shared
// transpose scratch buffer (every transpose call in this file operates on
// at most rows*columns elements — see the comment in svd_plan for why that
// bound holds for both the direct and swapped-dims paths, so one buffer of
// that size is reused sequentially throughout), and — only on the
// rows < columns path — the extra `at`/`u_prime`/`vt_prime` buffers
// svd_impl itself needs to factor A^T and relabel the result.
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
        // svd_core is invoked with dims swapped (columns, rows), so its own
        // k = min(columns, rows) = rows.
        p.core = svd_core_plan<T>(arena, columns, rows, rows, lwork);
        const auto rc = static_cast<size_t>(rows) * static_cast<size_t>(columns);
        p.at           = arena.next<T>(rc);
        p.u_prime       = arena.next<T>(static_cast<size_t>(columns) * static_cast<size_t>(rows));
        p.vt_prime      = arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(rows));
    }
    // Every transpose call in either path operates on a buffer of r'*c'
    // elements where r', c' <= max(rows, columns) and one of them is
    // min(rows, columns) — so r'*c' <= rows*columns always (k <= rows and
    // k <= columns for every k used as a dimension anywhere above).
    p.transpose_scratch =
        arena.next<T>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return p;
}

template <typename T, typename BufferSizeFn>
size_t svd_workspace_size_impl(BufferSizeFn buffer_size, linalg_long rows, linalg_long columns)
{
    LOGGING_CHECK(rows != 0 && columns != 0, "svd_decomposition_workspace_size: rows and columns must be positive");
    const auto r      = detail::checked_narrow<int>(rows);
    const auto c      = detail::checked_narrow<int>(columns);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto eff_r  = (r >= c) ? r : c;
    const auto eff_c  = (r >= c) ? c : r;
    int        lwork  = 0;
    LOGGING_CHECK(buffer_size(handle, eff_r, eff_c, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*gesvd_bufferSize failed");
    detail::workspace_arena arena;
    svd_plan<T>(arena, r, c, lwork);
    return arena.total_size();
}

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
// output via detail::transpose_device, the same identity used throughout
// lu_decomposition_gpu.cpp and inversion_gpu.cu: converting
// column-major storage of an (RX x CX) matrix to row-major storage of the
// same matrix is transpose_device(CX, RX, buf, scratch) — swapped
// dimensions relative to the matrix's own true shape. `bufs`/
// `transpose_scratch` are caller-owned scratch already carved out of the
// caller's workspace by svd_plan above.
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
    const auto k = std::min(r, c);  // == c, since the caller guarantees rows >= columns

    auto handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);
    const auto rk = static_cast<size_t>(r) * static_cast<size_t>(k);
    const auto kc = static_cast<size_t>(k) * static_cast<size_t>(c);

    LOGGING_CHECK(
        cudaMemcpyAsync(bufs.a_work, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    // See lu_decomposition_gpu.cpp: transposing the row-major buffer first
    // makes cuSOLVER's column-major (m=rows, n=columns, lda=rows) reading
    // of it equal A.
    detail::transpose_device(rows, columns, bufs.a_work, transpose_scratch, stream);

    const auto status = gesvd(handle, 'S', 'S', r, c, bufs.a_work, r, S, bufs.u_work, r,
        bufs.vt_work, k, bufs.work, lwork, bufs.rwork, info);
    LOGGING_CHECK(status == CUSOLVER_STATUS_SUCCESS, "cusolverDn*gesvd failed", static_cast<int>(status));

    LOGGING_CHECK(cudaMemcpyAsync(U, bufs.u_work, sizeof(T) * rk, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(static_cast<linalg_long>(k), rows, U, transpose_scratch, stream);

    LOGGING_CHECK(cudaMemcpyAsync(VT, bufs.vt_work, sizeof(T) * kc, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(columns, static_cast<linalg_long>(k), VT, transpose_scratch, stream);
}

// See svd_decomposition_gpu.h's RESTRICTION note: every caller-visible
// buffer here is required to be tightly packed row-major. `workspace`/
// `workspace_size` are caller-owned; keep them alive and unmodified until
// work queued on `stream` has completed.
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
    void*                  workspace,
    size_t                 workspace_size,
    cudaStream_t           stream)
{
    LOGGING_CHECK(A != nullptr && S != nullptr && U != nullptr && VT != nullptr && info != nullptr, "svd_decomposition: A, S, U, VT, and info must not be null");
    LOGGING_CHECK(rows != 0 && columns != 0, "svd_decomposition: rows and columns must be positive");

    const auto r = detail::checked_narrow<int>(rows);
    const auto c = detail::checked_narrow<int>(columns);
    const auto k = std::min(r, c);

    LOGGING_CHECK(!(detail::checked_narrow<int>(lda) != c) && !(detail::checked_narrow<int>(ldu) != k) && !(detail::checked_narrow<int>(ldv) != c), "linalg::gpu::svd_decomposition requires tightly packed lda/ldu/ldv "
                     "(lda == columns, ldu == min(rows, columns), ldv == columns)");

    auto       handle = detail::cusolver_handle_for_current_device();
    const auto eff_r   = (r >= c) ? r : c;
    const auto eff_c   = (r >= c) ? c : r;
    int        lwork   = 0;
    LOGGING_CHECK(buffer_size(handle, eff_r, eff_c, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*gesvd_bufferSize failed");

    LOGGING_CHECK(workspace_size >= svd_workspace_size_impl<T>(buffer_size, rows, columns),
        "svd_decomposition: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = svd_plan<T>(arena, r, c, lwork);

    if (r >= c)
    {
        svd_core(gesvd, rows, columns, A, S, U, VT, info, bufs.core, bufs.transpose_scratch, lwork,
            stream);
        return;
    }

    // cuSOLVER's classic gesvd only supports m >= n (see svd_core), so for
    // rows < columns this factors A^T (columns x rows, which does satisfy
    // m' >= n') instead and relabels: A = (A^T)^T = (U' S' VT')^T
    // = VT'^T S' U'^T, so this A's U = VT'^T (rows x k) and
    // VT = U'^T (k x columns); S = S' unchanged (SVD singular values are
    // transpose-invariant).
    const auto rc = static_cast<size_t>(r) * static_cast<size_t>(c);

    LOGGING_CHECK(
        cudaMemcpyAsync(bufs.at, A, sizeof(T) * rc, cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
        "cudaMemcpyAsync failed");
    // at becomes row-major (columns x rows) = A^T.
    detail::transpose_device(rows, columns, bufs.at, bufs.transpose_scratch, stream);

    const auto ck = static_cast<size_t>(c) * static_cast<size_t>(k);  // columns * k (k == rows)
    const auto kr = static_cast<size_t>(k) * static_cast<size_t>(r);  // k * rows

    svd_core(gesvd, columns, rows, bufs.at, S, bufs.u_prime, bufs.vt_prime, info, bufs.core,
        bufs.transpose_scratch, lwork, stream);

    // Caller's VT (k x columns) = transpose(u_prime (columns x k));
    // u_prime is already row-major (svd_core's own output convention), so
    // this is a genuine transpose, not the column-/row-major
    // reinterpretation trick used elsewhere in this file.
    LOGGING_CHECK(
        cudaMemcpyAsync(VT, bufs.u_prime, sizeof(T) * ck, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(columns, static_cast<linalg_long>(k), VT, bufs.transpose_scratch, stream);

    // Caller's U (rows x k) = transpose(vt_prime (k x rows)).
    LOGGING_CHECK(
        cudaMemcpyAsync(U, bufs.vt_prime, sizeof(T) * kr, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
    detail::transpose_device(static_cast<linalg_long>(k), rows, U, bufs.transpose_scratch, stream);
}

}  // namespace

size_t svd_decomposition_workspace_size(
    const float* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<float>(cusolverDnSgesvd_bufferSize, rows, columns);
}

size_t svd_decomposition_workspace_size(
    const double* /*type_tag*/, linalg_long rows, linalg_long columns)
{
    return svd_workspace_size_impl<double>(cusolverDnDgesvd_bufferSize, rows, columns);
}

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
    void*                            workspace,
    size_t                           workspace_size,
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
        workspace,
        workspace_size,
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
    void*                            workspace,
    size_t                           workspace_size,
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
        workspace,
        workspace_size,
        stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
