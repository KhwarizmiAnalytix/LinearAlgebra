#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"

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

cublasFillMode_t to_fill_mode(cholesky_decomposition_enum type)
{
    // cuSOLVER's potrf is column-major; a row-major LOWER-triangular factor
    // of a symmetric matrix occupies the same memory as the column-major
    // UPPER-triangular factor of the same matrix (and vice versa), so the
    // fill mode is flipped relative to the CPU (LAPACKE row-major) backend.
    return type == cholesky_decomposition_enum::LOWER_TRIANGULAR ? CUBLAS_FILL_MODE_UPPER
                                                                  : CUBLAS_FILL_MODE_LOWER;
}

}  // namespace

void cholesky_decomposition(float* C, quarisma_int lda, cholesky_decomposition_enum type, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();
    auto       fill   = to_fill_mode(type);

    int lwork = 0;
    if (cusolverDnSpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSpotrf_bufferSize failed");
    }

    float* workspace = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(float) * static_cast<size_t>(lwork));

    if (cusolverDnSpotrf(handle, fill, n, C, n, workspace, lwork, info) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        LINALG_THROW("cusolverDnSpotrf failed");
    }
    cudaFree(workspace);
}

void cholesky_decomposition(double* C, quarisma_int lda, cholesky_decomposition_enum type, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();
    auto       fill   = to_fill_mode(type);

    int lwork = 0;
    if (cusolverDnDpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDpotrf_bufferSize failed");
    }

    double* workspace = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(double) * static_cast<size_t>(lwork));

    if (cusolverDnDpotrf(handle, fill, n, C, n, workspace, lwork, info) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        LINALG_THROW("cusolverDnDpotrf failed");
    }
    cudaFree(workspace);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
