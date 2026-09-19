#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <vector>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation/linear_solver_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{
namespace
{

// `m`/`x` are device pointers (this is the cuda backend); `pivot` is host
// scratch, matching lu_decomposition_cublas.cxx's convention.
//
// cusolverDnXgetrf factors the column-major reading of the row-major buffer
// `m`, i.e. A^T rather than A (see lu_decomposition_cublas.cxx's caveat).
// Solving A x = b is then (A^T)^T x = b, i.e. cusolverDnXgetrs with
// trans = CUBLAS_OP_T against that same factorization — no extra transpose
// or copy needed.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
bool lu_solve_cuda(
    BufferSizeFn buffer_size, GetrfFn getrf, GetrsFn getrs, T* m, quarisma_int lda, T* x)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();

    int lwork = 0;
    if (buffer_size(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf_bufferSize failed");
    }

    T*   workspace = nullptr;
    int* dev_ipiv  = nullptr;
    int* dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_ipiv), sizeof(int) * static_cast<size_t>(n));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    auto status = getrf(handle, n, n, m, n, workspace, dev_ipiv, dev_info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = getrs(handle, CUBLAS_OP_T, n, 1, m, n, dev_ipiv, x, n, dev_info);
    }

    int host_info = -1;
    cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_ipiv);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*getrf/getrs failed");
    }
    return host_info == 0;
}

template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
bool cholesky_solve_cuda(
    BufferSizeFn buffer_size, PotrfFn potrf, PotrsFn potrs, T* m, quarisma_int lda, T* x)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();
    // Row-major LOWER factor <-> column-major UPPER fill, see
    // cholesky_decomposition_cublas.cxx.
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    T*   workspace = nullptr;
    int* dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    auto status = potrf(handle, fill, n, m, n, workspace, lwork, dev_info);
    if (status == CUSOLVER_STATUS_SUCCESS)
    {
        status = potrs(handle, fill, n, 1, m, n, x, n, dev_info);
    }

    int host_info = -1;
    cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf/potrs failed");
    }
    return host_info == 0;
}

}  // namespace

void solver_cublas_f32(float* m, quarisma_int* pivot, quarisma_int lda, float* x, linear_solver_type type)
{
    (void)pivot;  // cuSOLVER manages its own device-side pivots internally
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_solve_cuda(
                cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, lda, x);
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_solve_cuda(
                cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs, m, lda, x);
            break;
    }
}

void solver_cublas_f64(
    double* m, quarisma_int* pivot, quarisma_int lda, double* x, linear_solver_type type)
{
    (void)pivot;
    switch (type)
    {
        case linear_solver_type::LU_LINEAR_SOLVER:
        case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
            lu_solve_cuda(
                cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, lda, x);
            break;
        case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
        case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
            cholesky_solve_cuda(
                cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs, m, lda, x);
            break;
    }
}

LINALG_REGISTER_DISPATCH(solver_f32_stub, cuda, solver_cublas_f32);
LINALG_REGISTER_DISPATCH(solver_f64_stub, cuda, solver_cublas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
