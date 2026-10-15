#include "include/matrix_operation_gpu/linear_solver_gpu.h"

#if LINALG_HAS_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/common/cuda_workspace.h"
#include "include/matrix_operation_gpu/transpose_gpu.h"
#include <include/logging.h>
#include "include/common/narrow.h"

namespace linalg
{
namespace gpu
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
size_t lu_solve_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    int        lwork  = 0;
    LOGGING_CHECK(buffer_size(handle, n, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
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
size_t cholesky_solve_workspace_size(BufferSizeFn buffer_size, linalg_int lda)
{
    const auto n      = detail::checked_narrow<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    const auto fill   = CUBLAS_FILL_MODE_UPPER;
    int        lwork  = 0;
    LOGGING_CHECK(buffer_size(handle, fill, n, static_cast<T*>(nullptr), n, &lwork) == CUSOLVER_STATUS_SUCCESS,
        "cusolverDn*potrf_bufferSize failed");
    detail::workspace_arena arena;
    cholesky_solve_plan<T>(arena, lwork);
    return arena.total_size();
}

// See lu_decomposition_gpu.cpp: transposing `m` on the device first makes
// cuSOLVER's column-major reading of it equal the caller's actual A rather
// than A^T, so getrf below factors A itself and a plain (untransposed,
// CUBLAS_OP_N) getrs solves A x = b directly. Unlike lu_decomposition_gpu,
// `m` is pure internal scratch here — linear_solver never returns factor
// data to the caller — so there is no need to transpose it back afterward.
// `info` is the caller's device int*; the LU pivots themselves are internal
// device-side scratch cuSOLVER never exposes here. `workspace`/
// `workspace_size` are caller-owned; keep them alive and unmodified until
// work queued on `stream` has completed.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_solve(BufferSizeFn buffer_size,
    GetrfFn                getrf,
    GetrsFn                getrs,
    T*                     m,
    linalg_int           lda,
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
    LOGGING_CHECK(!(buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDn*getrf_bufferSize failed");

    LOGGING_CHECK(workspace_size >= lu_solve_workspace_size<T>(buffer_size, lda),
        "lu_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    const auto               bufs = lu_solve_plan<T>(arena, lwork, n);

    detail::transpose_device(lda, lda, m, bufs.transpose_scratch, stream);

    auto status = getrf(handle, n, n, m, n, bufs.workspace, bufs.dev_ipiv, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = getrs(handle, CUBLAS_OP_N, n, 1, m, n, bufs.dev_ipiv, x, n, info);
    }

    LOGGING_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDn*getrf/getrs failed");
}

template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_solve(BufferSizeFn buffer_size,
    PotrfFn                      potrf,
    PotrsFn                      potrs,
    T*                           m,
    linalg_int                 lda,
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
    // cholesky_decomposition_gpu.cpp.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    LOGGING_CHECK(!(buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDn*potrf_bufferSize failed");

    LOGGING_CHECK(workspace_size >= cholesky_solve_workspace_size<T>(buffer_size, lda),
        "cholesky_solve: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T*                       workspace_buf = cholesky_solve_plan<T>(arena, lwork);

    auto status = potrf(handle, fill, n, m, n, workspace_buf, lwork, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = potrs(handle, fill, n, 1, m, n, x, n, info);
    }

    LOGGING_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDn*potrf/potrs failed");
}

}  // namespace

size_t linear_solver_workspace_size(const float* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_solve_workspace_size<float>(cusolverDnSgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_solve_workspace_size<float>(cusolverDnSpotrf_bufferSize, lda);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

size_t linear_solver_workspace_size(const double* /*type_tag*/, linalg_int lda, linear_solver_type type)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_solve_workspace_size<double>(cusolverDnDgetrf_bufferSize, lda);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_solve_workspace_size<double>(cusolverDnDpotrf_bufferSize, lda);
    }
    LOGGING_THROW("unsupported linear_solver_type", static_cast<linalg_int>(type));
}

void linear_solver(float* m, linalg_int lda, float* x, linear_solver_type type, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream)
{
    LOGGING_CHECK(m != nullptr && x != nullptr && info != nullptr, "linear_solver: m, x, and info must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver: lda must be positive", lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, lda, x, info,
            workspace, workspace_size, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs, m, lda, x,
            info, workspace, workspace_size, stream);
        break;
    }
}

void linear_solver(double* m, linalg_int lda, double* x, linear_solver_type type, int* info,
    void* workspace, size_t workspace_size, cudaStream_t stream)
{
    LOGGING_CHECK(m != nullptr && x != nullptr && info != nullptr, "linear_solver: m, x, and info must not be null");
    LOGGING_CHECK(lda > 0, "linear_solver: lda must be positive", lda);
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, lda, x, info,
            workspace, workspace_size, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs, m, lda, x,
            info, workspace, workspace_size, stream);
        break;
    }
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_HAS_CUBLAS
