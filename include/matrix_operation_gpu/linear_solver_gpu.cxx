#include "include/matrix_operation_gpu/linear_solver_gpu.h"

#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{
namespace
{

// cusolverDnXgetrf factors the column-major reading of the row-major buffer
// `m`, i.e. A^T rather than A (see lu_decomposition_gpu.h's caveat). Solving
// A x = b is then (A^T)^T x = b, i.e. cusolverDnXgetrs with
// trans = CUBLAS_OP_T against that same factorization — no extra transpose
// or copy needed. `info` is the caller's device int*; the LU pivots
// themselves are internal device-side scratch cuSOLVER never exposes here.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_solve(
    BufferSizeFn buffer_size, GetrfFn getrf, GetrsFn getrs, T* m, quarisma_int lda, T* x, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf_bufferSize failed");
    }

    T*   workspace = nullptr;
    int* dev_ipiv  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_ipiv), sizeof(int) * static_cast<size_t>(n));

    auto status = getrf(handle, n, n, m, n, workspace, dev_ipiv, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = getrs(handle, CUBLAS_OP_T, n, 1, m, n, dev_ipiv, x, n, info);
    }

    cudaFree(workspace);
    cudaFree(dev_ipiv);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf/getrs failed");
    }
}

template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_solve(
    BufferSizeFn buffer_size, PotrfFn potrf, PotrsFn potrs, T* m, quarisma_int lda, T* x, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();
    // Row-major LOWER factor <-> column-major UPPER fill, see
    // cholesky_decomposition_gpu.cxx.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    T* workspace = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork));

    auto status = potrf(handle, fill, n, m, n, workspace, lwork, info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = potrs(handle, fill, n, 1, m, n, x, n, info);
    }

    cudaFree(workspace);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf/potrs failed");
    }
}

}  // namespace

void linear_solver(float* m, quarisma_int lda, float* x, linear_solver_type type, int* info)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_solve(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, lda, x, info);
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_solve(
                cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs, m, lda, x, info);
            break;
    }
}

void linear_solver(double* m, quarisma_int lda, double* x, linear_solver_type type, int* info)
{
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_solve(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, lda, x, info);
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_solve(
                cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs, m, lda, x, info);
            break;
    }
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
