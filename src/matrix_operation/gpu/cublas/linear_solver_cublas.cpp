#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/matrix_operation/linear_solver.h"
#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"
#include "include/common/workspace_arena.h"

// The GPU linear_solver manages pivots internally; the struct's `pivot`
// parameter is unused for this backend.
namespace linalg
{
namespace
{

template <typename T>
struct lu_solve_layout
{
    T*   workspace;
    int* dev_ipiv;
    T*   transpose_scratch;
};

template <typename T>
lu_solve_layout<T> lu_solve_plan(detail::workspace_arena& arena, int lwork, int n)
{
    lu_solve_layout<T> p;
    p.workspace         = arena.next<T>(static_cast<size_t>(lwork));
    p.dev_ipiv          = arena.next<int>(static_cast<size_t>(n));
    p.transpose_scratch = arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(n));
    return p;
}

template <typename T, typename BufferSizeFn>
size_t lu_solve_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*getrf_bufferSize failed");
    detail::workspace_arena arena;
    lu_solve_plan<T>(arena, lwork, n);
    return arena.total_size();
}

template <typename T>
T* cholesky_solve_plan(detail::workspace_arena& arena, int lwork)
{
    return arena.next<T>(static_cast<size_t>(lwork));
}

template <typename T, typename BufferSizeFn>
size_t cholesky_solve_workspace_size_impl(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LINALG_CHECK(
        buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_solve_plan<T>(arena, lwork);
    return arena.total_size();
}

// Returns the internal GPU workspace bytes (without the info slot) for a
// given scalar type and solver type.
template <typename T>
size_t gpu_ls_base(linalg_int lda, linear_solver_type type);

template <>
size_t gpu_ls_base<float>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_solve_workspace_size_impl<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_solve_workspace_size_impl<float>(cusolverDnSpotrf_bufferSize, lda);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

template <>
size_t gpu_ls_base<double>(linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_solve_workspace_size_impl<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_solve_workspace_size_impl<double>(cusolverDnDpotrf_bufferSize, lda);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

template <typename T>
size_t linear_solver_ws_size(linalg_int lda, linear_solver_type type)
{
    LINALG_CHECK(lda > 0, "linear_solver_workspace_size: lda must be positive", lda);
    const auto gpu_base = gpu_ls_base<T>(lda, type);
    detail::workspace_arena arena;
    arena.next<int>(1, alignof(int));
    arena.next<unsigned char>(gpu_base);
    return arena.total_size();
}

// See lu_decomposition_cublas.cpp: transposing `m` on the device first makes
// cuSOLVER's column-major reading of it equal the caller's actual A rather
// than A^T, so getrf below factors A itself and a plain (untransposed,
// CUBLAS_OP_N) getrs solves A x = b directly. Unlike lu_decomposition_cublas,
// `m` is pure internal scratch here — linear_solver never returns factor
// data to the caller — so there is no need to transpose it back afterward.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_solve(BufferSizeFn buffer_size,
    GetrfFn                getrf,
    GetrsFn                getrs,
    T*                     m,
    linalg_int             lda,
    T*                     x,
    int*                   info,
    void*                  workspace,
    size_t                 workspace_size,
    cudaStream_t           stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    LINALG_CHECK(!(buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS),
        "cusolverDn*getrf_bufferSize failed");

    LINALG_CHECK(workspace_size >= lu_solve_workspace_size_impl<T>(buffer_size, lda),
        "lu_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_solve_plan<T>(arena, lwork, n);

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = getrs(handle, CUBLAS_OP_N, n, 1, m, n, bufs.dev_ipiv, x, n, info);
    }

    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDn*getrf/getrs failed");
}

template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_solve(BufferSizeFn buffer_size,
    PotrfFn                      potrf,
    PotrsFn                      potrs,
    T*                           m,
    linalg_int                   lda,
    T*                           x,
    int*                         info,
    void*                        workspace,
    size_t                       workspace_size,
    cudaStream_t                 stream)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    // Row-major LOWER factor <-> column-major UPPER fill, see
    // cholesky_decomposition_cublas.cpp.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    LINALG_CHECK(!(buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS),
        "cusolverDn*potrf_bufferSize failed");

    LINALG_CHECK(workspace_size >= cholesky_solve_workspace_size_impl<T>(buffer_size, lda),
        "cholesky_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                       workspace_buf = cholesky_solve_plan<T>(arena, lwork);

    auto status = potrf(handle, fill, n, m, n, workspace_buf, lwork, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = potrs(handle, fill, n, 1, m, n, x, n, info);
    }

    LINALG_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDn*potrf/potrs failed");
}

template <typename T, typename BufSizeLuFn, typename GetrfFn, typename GetrsFn,
    typename BufSizeChFn, typename PotrfFn, typename PotrsFn>
void linear_solver_apply(BufSizeLuFn buf_size_lu,
    GetrfFn     getrf,
    GetrsFn     getrs,
    BufSizeChFn buf_size_ch,
    PotrfFn     potrf,
    PotrsFn     potrs,
    T*          m,
    linalg_int  lda,
    T*          x,
    linear_solver_type type,
    void*       workspace,
    size_t      workspace_size,
    void*       stream)
{
    LINALG_CHECK(m != nullptr && x != nullptr, "linear_solver: m and x must not be null");
    LINALG_CHECK(lda > 0, "linear_solver: lda must be positive", lda);
    const auto stream_t = static_cast<cudaStream_t>(stream);

    // Workspace layout: one int (dev_info) first, then the cuSOLVER buffer.
    // The total was computed by linear_solver_ws_size<T> and includes both.
    const auto gpu_base = gpu_ls_base<T>(lda, type);
    detail::workspace_arena arena(workspace, workspace_size);
    int*  dev_info = arena.next<int>(1, alignof(int));
    void* gpu_ws   = arena.next<unsigned char>(gpu_base);

    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(buf_size_lu, getrf, getrs, m, lda, x, dev_info, gpu_ws, gpu_base, stream_t);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(buf_size_ch, potrf, potrs, m, lda, x, dev_info, gpu_ws, gpu_base,
            stream_t);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
    }
}

}  // namespace

template <>
void linear_solver<float, linalg_backend::CUDA_BLAS>::apply(float* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    float*             x,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("linear_solver/cuda_blas<float>");
    linear_solver_apply(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs,
        cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs,
        m, lda, x, type, workspace, workspace_size, stream);
}

template <>
void linear_solver<double, linalg_backend::CUDA_BLAS>::apply(double* m,
    linalg_int*        /*pivot*/,
    linalg_int         lda,
    double*            x,
    linear_solver_type type,
    void*              workspace,
    size_t             workspace_size,
    void*              stream)
{
    LINALG_SCOPE("linear_solver/cuda_blas<double>");
    linear_solver_apply(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs,
        cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs,
        m, lda, x, type, workspace, workspace_size, stream);
}

template <>
size_t linear_solver<float, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return linear_solver_ws_size<float>(lda, type);
}

template <>
size_t linear_solver<double, linalg_backend::CUDA_BLAS>::workspace_size(
    linalg_int lda, linear_solver_type type)
{
    return linear_solver_ws_size<double>(lda, type);
}

}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
