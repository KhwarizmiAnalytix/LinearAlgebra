#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation/cholesky_decomposition_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{
namespace
{

cublasFillMode_t to_fill_mode(cholesky_decomposition_enum type)
{
    // cuSOLVER's potrf is column-major; a row-major LOWER-triangular factor
    // of a symmetric matrix occupies the same memory as the column-major
    // UPPER-triangular factor of the same matrix (and vice versa), so the
    // fill mode is flipped relative to the CPU (LAPACKE row-major) backends.
    return type == cholesky_decomposition_enum::LOWER_TRIANGULAR ? CUBLAS_FILL_MODE_UPPER
                                                                  : CUBLAS_FILL_MODE_LOWER;
}

int device_info(cudaError_t copy_status, int host_info)
{
    if (copy_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpy of cuSOLVER devInfo failed");
    }
    return host_info;
}

}  // namespace

bool cholesky_cublas_f32(float* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();
    auto       fill   = to_fill_mode(type);

    int lwork = 0;
    if (cusolverDnSpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSpotrf_bufferSize failed");
    }

    float* workspace = nullptr;
    int*   dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(float) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    const auto status =
        cusolverDnSpotrf(handle, fill, n, C, n, workspace, lwork, dev_info);

    int host_info = -1;
    const auto copy_status =
        cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSpotrf failed");
    }
    return device_info(copy_status, host_info) == 0;
}

bool cholesky_cublas_f64(double* C, quarisma_int lda, cholesky_decomposition_enum type)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();
    auto       fill   = to_fill_mode(type);

    int lwork = 0;
    if (cusolverDnDpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDpotrf_bufferSize failed");
    }

    double* workspace = nullptr;
    int*    dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(double) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    const auto status =
        cusolverDnDpotrf(handle, fill, n, C, n, workspace, lwork, dev_info);

    int host_info = -1;
    const auto copy_status =
        cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDpotrf failed");
    }
    return device_info(copy_status, host_info) == 0;
}

LINALG_REGISTER_DISPATCH(cholesky_f32_stub, cuda, cholesky_cublas_f32);
LINALG_REGISTER_DISPATCH(cholesky_f64_stub, cuda, cholesky_cublas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
