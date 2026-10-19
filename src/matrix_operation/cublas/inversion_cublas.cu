#if LINALG_HAS_CUBLAS

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/inversion.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"
#include "include/common/workspace_arena.h"

// File-scope (not namespaced) __global__ templates: only ever launched from
// this translation unit (same pattern as inversion_gpu.cu).

// Identity is symmetric, so a flat row-major fill is correct whichever way
// the result is later read (row- or column-major).
template <typename scalar_t>
static __global__ void fill_identity_kernel(int n, scalar_t* out)
{
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || col >= n)
    {
        return;
    }
    out[row * n + col] = (row == col) ? scalar_t(1) : scalar_t(0);
}

// Single-thread diagonal product with pivot sign. `m` is the row-major-packed
// L\U factor lu_decomposition-style code produces. `pivot` is cuSOLVER's own
// 1-based row-swap sequence: each pivot[i] != i+1 entry flips the sign.
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
namespace
{

dim3 square_grid(int n, dim3 block)
{
    return dim3((static_cast<unsigned>(n) + block.x - 1) / block.x,
        (static_cast<unsigned>(n) + block.y - 1) / block.y);
}

// ---- Inversion buffer layouts ----

template <typename T>
struct lu_invert_layout
{
    T*   workspace;
    int* dev_ipiv;
    T*   identity;
    T*   transpose_scratch;
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

template <typename T, typename BufferSizeFn>
size_t lu_invert_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_invert_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T, typename BufferSizeFn>
size_t cholesky_invert_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_invert_plan<T>(arena, lwork, n);
    return arena.total_size();
}

// ---- Determinant buffer layouts ----

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
size_t lu_determinant_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_determinant_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T, typename BufferSizeFn>
size_t cholesky_determinant_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_determinant_plan<T>(arena, lwork);
    return arena.total_size();
}

// Returns the internal GPU workspace size for a given inversion type.
// The struct workspace_size() adds an info slot on top.
template <typename T>
size_t gpu_invert_base(linalg_int lda, linear_solver_type type);

template <>
size_t gpu_invert_base<float>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_invert_workspace_size_impl<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_invert_workspace_size_impl<float>(cusolverDnSpotrf_bufferSize, lda);
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <>
size_t gpu_invert_base<double>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_invert_workspace_size_impl<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_invert_workspace_size_impl<double>(cusolverDnDpotrf_bufferSize, lda);
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <typename T>
size_t gpu_determinant_base(linalg_int lda, linear_solver_type type);

template <>
size_t gpu_determinant_base<float>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant_workspace_size_impl<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant_workspace_size_impl<float>(cusolverDnSpotrf_bufferSize, lda);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

template <>
size_t gpu_determinant_base<double>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant_workspace_size_impl<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant_workspace_size_impl<double>(cusolverDnDpotrf_bufferSize, lda);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

template <typename T>
size_t inversion_ws_size(linalg_int lda, linear_solver_type type)
{
    LINALG_CHECK(lda > 0, "inversion_workspace_size: lda must be positive", lda);
    const auto gpu_base = gpu_invert_base<T>(lda, type);
    detail::workspace_arena arena;
    arena.next<int>(1, alignof(int));
    arena.next<unsigned char>(gpu_base);
    return arena.total_size();
}

template <typename T>
size_t determinant_ws_size(linalg_int lda, linear_solver_type type)
{
    LINALG_CHECK(lda > 0, "determinant_workspace_size: lda must be positive", lda);
    return gpu_determinant_base<T>(lda, type);
}

// LU-based in-place inversion.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_invert(BufferSizeFn buffer_size,
    GetrfFn                 getrf,
    GetrsFn                 getrs,
    T*                      m,
    linalg_int              lda,
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
        LINALG_THROW("cusolverDn*getrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    lu_invert_plan<T>(sizing, lwork, n);
    LINALG_CHECK(workspace_size >= sizing.total_size(), "lu_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_invert_plan<T>(arena, lwork, n);

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, bufs.identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        LINALG_THROW("fill_identity_kernel launch failed");
    }

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf failed");
    }

    const auto solve_status =
        getrs(handle, CUBLAS_OP_N, n, n, m, n, bufs.dev_ipiv, bufs.identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrs failed");
    }

    // `identity` now holds A^-1 column-major; transpose to row-major, then
    // copy over the caller's buffer in place.
    detail::transpose_device(lda, lda, bufs.identity, bufs.transpose_scratch, stream);
    const auto nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    LINALG_CHECK(
        cudaMemcpyAsync(m, bufs.identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
}

// Cholesky-based in-place inversion.
template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_invert(BufferSizeFn buffer_size,
    PotrfFn                       potrf,
    PotrsFn                       potrs,
    T*                            m,
    linalg_int                    lda,
    int*                          info,
    void*                         workspace,
    size_t                        workspace_size,
    cudaStream_t                  stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    cholesky_invert_plan<T>(sizing, lwork, n);
    LINALG_CHECK(workspace_size >= sizing.total_size(),
        "cholesky_invert: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = cholesky_invert_plan<T>(arena, lwork, n);

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, bufs.identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        LINALG_THROW("fill_identity_kernel launch failed");
    }

    const auto status = potrf(handle, fill, n, m, n, bufs.workspace, lwork, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf failed");
    }

    const auto solve_status = potrs(handle, fill, n, n, m, n, bufs.identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrs failed");
    }

    detail::transpose_device(lda, lda, bufs.identity, bufs.transpose_scratch, stream);
    const auto nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    LINALG_CHECK(
        cudaMemcpyAsync(m, bufs.identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream) ==
            cudaSuccess,
        "cudaMemcpyAsync failed");
}

template <typename T, typename BufferSizeFn, typename GetrfFn>
T lu_determinant(BufferSizeFn buffer_size,
    GetrfFn                   getrf,
    T*                        m,
    linalg_int                lda,
    void*                     workspace,
    size_t                    workspace_size,
    cudaStream_t              stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    lu_determinant_plan<T>(sizing, lwork, n);
    LINALG_CHECK(workspace_size >= sizing.total_size(),
        "lu_determinant: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_determinant_plan<T>(arena, lwork, n);

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    const auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, bufs.dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf failed");
    }

    lu_determinant_kernel<T><<<1, 1, 0, stream>>>(m, bufs.dev_ipiv, n, bufs.dev_det);
    const auto launch_err = cudaGetLastError();

    T          det         = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, bufs.dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    if (launch_err != cudaSuccess)
    {
        LINALG_THROW("lu_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

template <typename T, typename BufferSizeFn, typename PotrfFn>
T cholesky_determinant(BufferSizeFn buffer_size,
    PotrfFn                         potrf,
    T*                              m,
    linalg_int                      lda,
    void*                           workspace,
    size_t                          workspace_size,
    cudaStream_t                    stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    detail::workspace_arena sizing;
    cholesky_determinant_plan<T>(sizing, lwork);
    LINALG_CHECK(workspace_size >= sizing.total_size(),
        "cholesky_determinant: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = cholesky_determinant_plan<T>(arena, lwork);

    const auto status = potrf(handle, fill, n, m, n, bufs.workspace, lwork, bufs.dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf failed");
    }

    cholesky_determinant_kernel<T><<<1, 1, 0, stream>>>(m, n, bufs.dev_det);
    const auto launch_err = cudaGetLastError();

    T          det         = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, bufs.dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    if (launch_err != cudaSuccess)
    {
        LINALG_THROW("cholesky_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

template <typename T, typename BufSizeLuFn, typename GetrfFn, typename GetrsFn,
    typename BufSizeChFn, typename PotrfFn, typename PotrsFn>
void inversion_apply(BufSizeLuFn buf_size_lu,
    GetrfFn     getrf,
    GetrsFn     getrs,
    BufSizeChFn buf_size_ch,
    PotrfFn     potrf,
    PotrsFn     potrs,
    T*          m,
    linalg_int  lda,
    linear_solver_type type,
    void*       workspace,
    size_t      workspace_size,
    void*       stream)
{
    if (m == nullptr)
    {
        LINALG_THROW("inversion: m must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("inversion: lda must be positive", lda);
    }
    const auto stream_t = static_cast<cudaStream_t>(stream);
    const auto gpu_base = gpu_invert_base<T>(lda, type);
    detail::workspace_arena arena(workspace, workspace_size);
    int*  dev_info = arena.next<int>(1, alignof(int));
    void* gpu_ws   = arena.next<unsigned char>(gpu_base);

    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_invert(buf_size_lu, getrf, getrs, m, lda, dev_info, gpu_ws, gpu_base, stream_t);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_invert(buf_size_ch, potrf, potrs, m, lda, dev_info, gpu_ws, gpu_base,
            stream_t);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

template <typename T, typename BufSizeLuFn, typename GetrfFn, typename BufSizeChFn,
    typename PotrfFn>
T determinant_apply(BufSizeLuFn buf_size_lu,
    GetrfFn     getrf,
    BufSizeChFn buf_size_ch,
    PotrfFn     potrf,
    T*          m,
    linalg_int  lda,
    linear_solver_type type,
    void*       workspace,
    size_t      workspace_size,
    void*       stream)
{
    if (m == nullptr)
    {
        LINALG_THROW("determinant: m must not be null");
    }
    if (lda <= 0)
    {
        LINALG_THROW("determinant: lda must be positive", lda);
    }
    const auto stream_t = static_cast<cudaStream_t>(stream);

    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(buf_size_lu, getrf, m, lda, workspace, workspace_size, stream_t);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(buf_size_ch, potrf, m, lda, workspace, workspace_size,
            stream_t);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

}  // namespace

// ---- inversion<T, cuda_blas> specializations ----

template <>
void inversion<float, linalg_backend::cuda_blas>::apply(float* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("inversion/cuda_blas<float>");
    inversion_apply(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs,
        cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs,
        m, lda, type, workspace, workspace_size, stream);
}

template <>
void inversion<double, linalg_backend::cuda_blas>::apply(double* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("inversion/cuda_blas<double>");
    inversion_apply(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs,
        cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs,
        m, lda, type, workspace, workspace_size, stream);
}

template <>
size_t inversion<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return inversion_ws_size<float>(lda, type);
}

template <>
size_t inversion<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return inversion_ws_size<double>(lda, type);
}

// ---- determinant<T, cuda_blas> specializations ----

template <>
float determinant<float, linalg_backend::cuda_blas>::apply(float* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("determinant/cuda_blas<float>");
    return determinant_apply(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf,
        cusolverDnSpotrf_bufferSize, cusolverDnSpotrf,
        m, lda, type, workspace, workspace_size, stream);
}

template <>
double determinant<double, linalg_backend::cuda_blas>::apply(double* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("determinant/cuda_blas<double>");
    return determinant_apply(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf,
        cusolverDnDpotrf_bufferSize, cusolverDnDpotrf,
        m, lda, type, workspace, workspace_size, stream);
}

template <>
size_t determinant<float, linalg_backend::cuda_blas>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return determinant_ws_size<float>(lda, type);
}

template <>
size_t determinant<double, linalg_backend::cuda_blas>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return determinant_ws_size<double>(lda, type);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
