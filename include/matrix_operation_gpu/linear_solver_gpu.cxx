#include "include/matrix_operation_gpu/linear_solver_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

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

// See lu_decomposition_gpu.cxx: transposing `m` on the device first makes
// cuSOLVER's column-major reading of it equal the caller's actual A rather
// than A^T, so getrf below factors A itself and a plain (untransposed,
// CUBLAS_OP_N) getrs solves A x = b directly. Unlike lu_decomposition_gpu,
// `m` is pure internal scratch here — linear_solver never returns factor
// data to the caller — so there is no need to transpose it back afterward.
// `info` is the caller's device int*; the LU pivots themselves are internal
// device-side scratch cuSOLVER never exposes here.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_solve(BufferSizeFn buffer_size,
    GetrfFn                getrf,
    GetrsFn                getrs,
    T*                     m,
    quarisma_int           lda,
    T*                     x,
    int*                   info,
    cudaStream_t           stream)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf_bufferSize failed");
    }

    T*   workspace = nullptr;
    int* dev_ipiv  = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ipiv), sizeof(int) * static_cast<size_t>(n)) !=
        cudaSuccess)
    {
        cudaFree(workspace);
        LINALG_THROW("cudaMalloc failed");
    }

    matrix_transpose(lda, lda, m, stream);

    auto status = getrf(handle, n, n, m, n, workspace, dev_ipiv, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = getrs(handle, CUBLAS_OP_N, n, 1, m, n, dev_ipiv, x, n, info);
    }

    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);
    cudaFree(dev_ipiv);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf/getrs failed");
    }
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }
}

template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_solve(BufferSizeFn buffer_size,
    PotrfFn                      potrf,
    PotrsFn                      potrs,
    T*                           m,
    quarisma_int                 lda,
    T*                           x,
    int*                         info,
    cudaStream_t                 stream)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    // Row-major LOWER factor <-> column-major UPPER fill, see
    // cholesky_decomposition_gpu.cxx.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    T* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }

    auto status = potrf(handle, fill, n, m, n, workspace, lwork, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = potrs(handle, fill, n, 1, m, n, x, n, info);
    }

    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf/potrs failed");
    }
    if (sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaStreamSynchronize failed");
    }
}

}  // namespace

void linear_solver(
    float* m, quarisma_int lda, float* x, linear_solver_type type, int* info, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(cusolverDnSgetrf_bufferSize,
            cusolverDnSgetrf,
            cusolverDnSgetrs,
            m,
            lda,
            x,
            info,
            stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(cusolverDnSpotrf_bufferSize,
            cusolverDnSpotrf,
            cusolverDnSpotrs,
            m,
            lda,
            x,
            info,
            stream);
        break;
    }
}

void linear_solver(
    double* m, quarisma_int lda, double* x, linear_solver_type type, int* info, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_solve(cusolverDnDgetrf_bufferSize,
            cusolverDnDgetrf,
            cusolverDnDgetrs,
            m,
            lda,
            x,
            info,
            stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_solve(cusolverDnDpotrf_bufferSize,
            cusolverDnDpotrf,
            cusolverDnDpotrs,
            m,
            lda,
            x,
            info,
            stream);
        break;
    }
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
