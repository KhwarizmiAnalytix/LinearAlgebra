#include "include/matrix_operation_gpu/inversion_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include <include/logging.h>
#include "include/common/narrow.h"

// File-scope (not namespaced) __global__ templates, matching the usual CUDA
// idiom already used by multiplication_batched_gpu.cu for a kernel
// only ever launched from its own translation unit.

// Identity is symmetric, so a flat row-major fill is correct whichever way
// the result is later read (row- or column-major).
template <typename scalar_t> static __global__ void fill_identity_kernel(int n, scalar_t* out)
{
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || col >= n)
    {
        return;
    }
    out[row * n + col] = (row == col) ? scalar_t(1) : scalar_t(0);
}

// Single-thread reduction: n is one matrix dimension, not a large array, so
// a parallel reduction buys nothing here. `m` is the row-major-packed L\U
// factor lu_decomposition_gpu-style code produces (see lu_decomposition_gpu
// .cpp) — the diagonal offset i*n+i is the same whether m is read row- or
// column-major, so this needs no transpose either way. `pivot` is
// cuSOLVER's own 1-based row-swap sequence for A (already verified to match
// the CPU lapacke_?getrf convention once `m` has been transposed before
// getrf — see lu_decomposition_gpu.cpp): each pivot[i] != i+1 entry flips
// the sign, matching lu_determinant's caller (matrix_determinant_helper) in
// src/matrix_operation/inversion.cpp's CPU counterpart.
template <typename scalar_t>
static __global__ void lu_determinant_kernel(
    const scalar_t* m, const int* pivot, int n, scalar_t* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    scalar_t det    = scalar_t(1);
    bool     negate = false;
    for (int i = 0; i < n; ++i)
    {
        det *= m[i * n + i];
        if (pivot[i] - 1 != i)
        {
            negate = !negate;
        }
    }
    *out = negate ? -det : det;
}

// Cholesky has no pivoting: det(A) = det(L)^2 = product(diag(L))^2.
template <typename scalar_t>
static __global__ void cholesky_determinant_kernel(const scalar_t* m, int n, scalar_t* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    scalar_t det = scalar_t(1);
    for (int i = 0; i < n; ++i)
    {
        det *= m[i * n + i];
    }
    *out = det * det;
}

namespace linalg
{
namespace gpu
{
namespace
{

dim3 square_grid(int n, dim3 block)
{
    return dim3((static_cast<unsigned>(n) + block.x - 1) / block.x,
        (static_cast<unsigned>(n) + block.y - 1) / block.y);
}

// Buffer layouts shared between each operation's `*_workspace_size()` query
// and its real call, via detail::workspace_arena (include/common/
// cuda_workspace.h) — see that header for why one `_plan` function serving
// both a sizing-mode and a real arena is used instead of two hand-written,
// driftable buffer lists.
template <typename T>
struct lu_invert_layout
{
    T*   workspace;
    int* dev_ipiv;
    T*   identity;
    T*   transpose_scratch;  // reused for both the pre- and post-getrf transpose
};

template <typename T>
lu_invert_layout<T> lu_invert_plan(detail::workspace_arena& arena, int lwork, int n)
{
    lu_invert_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.dev_ipiv          = arena.next<int>(static_cast<size_t>(n));
    p.identity          = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T>
struct cholesky_invert_layout
{
    T* workspace;
    T* identity;
    T* transpose_scratch;
};

template <typename T>
cholesky_invert_layout<T> cholesky_invert_plan(detail::workspace_arena& arena, int lwork, int n)
{
    cholesky_invert_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.identity          = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T>
struct lu_determinant_layout
{
    T*   workspace;
    int* dev_ipiv;
    int* dev_info;
    T*   dev_det;
    T*   transpose_scratch;
};

template <typename T>
lu_determinant_layout<T> lu_determinant_plan(detail::workspace_arena& arena, int lwork, int n)
{
    lu_determinant_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.dev_ipiv          = arena.next<int>(static_cast<size_t>(n));
    p.dev_info          = arena.next<int>(1);
    p.dev_det           = arena.next<T>(1);
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T>
struct cholesky_determinant_layout
{
    T*   workspace;
    int* dev_info;
    T*   dev_det;
};

template <typename T>
cholesky_determinant_layout<T> cholesky_determinant_plan(detail::workspace_arena& arena, int lwork)
{
    cholesky_determinant_layout<T> p;
    p.workspace = arena.next<T>(static_cast<size_t>(lwork));
    p.dev_info  = arena.next<int>(1);
    p.dev_det   = arena.next<T>(1);
    return p;
}

template <typename T, typename BufferSizeFn>
size_t lu_invert_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LOGGING_CHECK(buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_invert_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T, typename BufferSizeFn>
size_t cholesky_invert_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill    = CUBLAS_FILL_MODE_UPPER;
    int        lwork   = 0;
    LOGGING_CHECK(buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_invert_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T, typename BufferSizeFn>
size_t lu_determinant_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LOGGING_CHECK(buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_determinant_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T, typename BufferSizeFn>
size_t cholesky_determinant_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill    = CUBLAS_FILL_MODE_UPPER;
    int        lwork   = 0;
    LOGGING_CHECK(buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_determinant_plan<T>(arena, lwork);
    return arena.total_size();
}

// LU-based in-place inversion: factor A (via the same transpose-getrf
// trick as lu_decomposition_gpu.cpp), solve A X = I for X = A^-1 against an
// on-device identity (nrhs = n), transpose the column-major solution back
// to row-major, and copy it over the caller's buffer in place — mirroring
// the CPU matrix_invert's factor-then-solve-against-identity strategy
// (src/matrix_operation/inversion.cpp) exactly, just with an
// explicit identity matrix standing in for LAPACKE's getri. `workspace`/
// `workspace_size` are caller-owned (see lu_invert_workspace_size above);
// the caller must keep them alive and unmodified until work queued on
// `stream` by this call has completed — nothing here synchronizes the
// stream for them.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_invert(BufferSizeFn buffer_size,
    GetrfFn                 getrf,
    GetrsFn                 getrs,
    T*                      m,
    linalg_int            lda,
    int*                    info,
    void*                   workspace,
    size_t                  workspace_size,
    cudaStream_t            stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*getrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    lu_invert_plan<T>(sizing, lwork, n);
    LOGGING_CHECK(workspace_size >= sizing.total_size(), "lu_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_invert_plan<T>(arena, lwork, n);

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, bufs.identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        LOGGING_THROW("fill_identity_kernel launch failed");
    }

    // See lu_decomposition_gpu.cpp: transpose first so getrf factors A, not
    // A^T.
    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*getrf failed");
    }

    const auto solve_status =
        getrs(handle, CUBLAS_OP_N, n, n, m, n, bufs.dev_ipiv, bufs.identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*getrs failed");
    }

    // `identity` now holds A^-1 column-major; transpose to row-major, then
    // copy over the caller's buffer in place.
    detail::transpose_device(lda, lda, bufs.identity, bufs.transpose_scratch, stream);
    const auto nn          = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto copy_status =
        cudaMemcpyAsync(m, bufs.identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream);
    if (copy_status != cudaSuccess)
    {
        LOGGING_THROW("cudaMemcpyAsync failed");
    }
}

// Cholesky-based in-place inversion: A is symmetric, so — unlike the LU
// path — no transpose is needed around potrf (see cholesky_decomposition_
// gpu.cpp); only the solution of A X = I still needs the transpose-back
// treatment, since X is not itself symmetric in general.
template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_invert(BufferSizeFn buffer_size,
    PotrfFn                       potrf,
    PotrsFn                       potrs,
    T*                            m,
    linalg_int                  lda,
    int*                          info,
    void*                         workspace,
    size_t                        workspace_size,
    cudaStream_t                  stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    // Row-major LOWER factor <-> column-major UPPER fill, see
    // cholesky_decomposition_gpu.cpp.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*potrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    cholesky_invert_plan<T>(sizing, lwork, n);
    LOGGING_CHECK(workspace_size >= sizing.total_size(), "cholesky_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = cholesky_invert_plan<T>(arena, lwork, n);

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, bufs.identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        LOGGING_THROW("fill_identity_kernel launch failed");
    }

    const auto status = potrf(handle, fill, n, m, n, bufs.workspace, lwork, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*potrf failed");
    }

    const auto solve_status = potrs(handle, fill, n, n, m, n, bufs.identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*potrs failed");
    }

    detail::transpose_device(lda, lda, bufs.identity, bufs.transpose_scratch, stream);
    const auto nn          = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto copy_status =
        cudaMemcpyAsync(m, bufs.identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream);
    if (copy_status != cudaSuccess)
    {
        LOGGING_THROW("cudaMemcpyAsync failed");
    }
}

// `info` is internal scratch here (unlike matrix_invert, determinant
// exposes no info/pivot to the caller at all — the only observable output
// is the returned scalar): a structurally failed getrf/potrf call (bad
// arguments, workspace too small, ...) throws, but a numerically singular
// matrix (cuSOLVER's info > 0, factorization completed with a zero pivot)
// is not distinguished from a healthy call — its diagonal product is
// already ~0 in that case, so the returned determinant naturally matches
// the CPU determinant's silent-zero-on-singular behavior
// (matrix_determinant_helper in inversion.cpp) without extra
// bookkeeping.
template <typename T, typename BufferSizeFn, typename GetrfFn>
T lu_determinant(BufferSizeFn buffer_size,
    GetrfFn                   getrf,
    T*                        m,
    linalg_int              lda,
    void*                    workspace,
    size_t                   workspace_size,
    cudaStream_t             stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*getrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    lu_determinant_plan<T>(sizing, lwork, n);
    LOGGING_CHECK(workspace_size >= sizing.total_size(), "lu_determinant: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_determinant_plan<T>(arena, lwork, n);

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, bufs.dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*getrf failed");
    }

    lu_determinant_kernel<T><<<1, 1, 0, stream>>>(m, bufs.dev_ipiv, n, bufs.dev_det);
    const auto launch_err = cudaGetLastError();

    T          det = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, bufs.dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    if (launch_err != cudaSuccess)
    {
        LOGGING_THROW("lu_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

template <typename T, typename BufferSizeFn, typename PotrfFn>
T cholesky_determinant(BufferSizeFn buffer_size,
    PotrfFn                         potrf,
    T*                              m,
    linalg_int                    lda,
    void*                          workspace,
    size_t                         workspace_size,
    cudaStream_t                   stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*potrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    cholesky_determinant_plan<T>(sizing, lwork);
    LOGGING_CHECK(
        workspace_size >= sizing.total_size(), "cholesky_determinant: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = cholesky_determinant_plan<T>(arena, lwork);

    const auto status = potrf(handle, fill, n, m, n, bufs.workspace, lwork, bufs.dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LOGGING_THROW("cusolverDn*potrf failed");
    }

    cholesky_determinant_kernel<T><<<1, 1, 0, stream>>>(m, n, bufs.dev_det);
    const auto launch_err = cudaGetLastError();

    T          det = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, bufs.dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    if (launch_err != cudaSuccess)
    {
        LOGGING_THROW("cholesky_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

}  // namespace

size_t matrix_invert_workspace_size(const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_invert_workspace_size<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_invert_workspace_size<float>(cusolverDnSpotrf_bufferSize, lda);
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

size_t matrix_invert_workspace_size(const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_invert_workspace_size<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_invert_workspace_size<double>(cusolverDnDpotrf_bufferSize, lda);
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

void matrix_invert(float* m, linalg_int lda, int* info, linear_solver_type type, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    if (m == nullptr || info == nullptr)
    {
        LOGGING_THROW("matrix_invert: m and info must not be null");
    }
    if (lda <= 0)
    {
        LOGGING_THROW("matrix_invert: lda must be positive", lda);
    }
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_invert(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, lda, info,
            workspace, workspace_size, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_invert(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs, m, lda,
            info, workspace, workspace_size, stream);
        break;
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

void matrix_invert(double* m, linalg_int lda, int* info, linear_solver_type type, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    if (m == nullptr || info == nullptr)
    {
        LOGGING_THROW("matrix_invert: m and info must not be null");
    }
    if (lda <= 0)
    {
        LOGGING_THROW("matrix_invert: lda must be positive", lda);
    }
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_invert(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, lda, info,
            workspace, workspace_size, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_invert(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs, m, lda,
            info, workspace, workspace_size, stream);
        break;
    default:
        LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

size_t matrix_determinant_workspace_size(
    const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant_workspace_size<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant_workspace_size<float>(cusolverDnSpotrf_bufferSize, lda);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

size_t matrix_determinant_workspace_size(
    const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant_workspace_size<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant_workspace_size<double>(cusolverDnDpotrf_bufferSize, lda);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

float determinant(float* m, linalg_int lda, linear_solver_type type, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    if (m == nullptr)
    {
        LOGGING_THROW("determinant: m must not be null");
    }
    if (lda <= 0)
    {
        LOGGING_THROW("determinant: lda must be positive", lda);
    }
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(
            cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, m, lda, workspace, workspace_size, stream);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, m, lda,
            workspace, workspace_size, stream);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

double determinant(double* m, linalg_int lda, linear_solver_type type, void* workspace,
    size_t workspace_size, cudaStream_t stream)
{
    if (m == nullptr)
    {
        LOGGING_THROW("determinant: m must not be null");
    }
    if (lda <= 0)
    {
        LOGGING_THROW("determinant: lda must be positive", lda);
    }
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(
            cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, m, lda, workspace, workspace_size, stream);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, m, lda,
            workspace, workspace_size, stream);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
