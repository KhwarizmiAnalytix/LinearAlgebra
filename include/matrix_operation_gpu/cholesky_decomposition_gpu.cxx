#include "include/matrix_operation_gpu/cholesky_decomposition_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "ThirdParty/Logging/include/logging.h"

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

void cholesky_decomposition(
    float* C, linalg_int lda, cholesky_decomposition_enum type, int* info, cudaStream_t stream)
{
    LOGGING_CHECK(C != nullptr && info != nullptr, "cholesky_decomposition: C and info must not be null");
    LOGGING_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    auto fill = to_fill_mode(type);

    int lwork = 0;
    LOGGING_CHECK(!(cusolverDnSpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnSpotrf_bufferSize failed");

    float* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
            sizeof(float) * static_cast<size_t>(lwork)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }

    const auto  status      = cusolverDnSpotrf(handle, fill, n, C, n, workspace, lwork, info);
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);
    LOGGING_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnSpotrf failed");
    LOGGING_CHECK(!(sync_status != cudaSuccess), "cudaStreamSynchronize failed");
}

void cholesky_decomposition(
    double* C, linalg_int lda, cholesky_decomposition_enum type, int* info, cudaStream_t stream)
{
    LOGGING_CHECK(C != nullptr && info != nullptr, "cholesky_decomposition: C and info must not be null");
    LOGGING_CHECK(lda > 0, "cholesky_decomposition: lda must be positive", lda);
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    auto fill = to_fill_mode(type);

    int lwork = 0;
    LOGGING_CHECK(!(cusolverDnDpotrf_bufferSize(handle, fill, n, C, n, &lwork) != CUSOLVER_STATUS_SUCCESS), "cusolverDnDpotrf_bufferSize failed");

    double* workspace = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace),
            sizeof(double) * static_cast<size_t>(lwork)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }

    const auto  status      = cusolverDnDpotrf(handle, fill, n, C, n, workspace, lwork, info);
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(workspace, stream, &sync_status);
    LOGGING_CHECK(!(status != CUSOLVER_STATUS_SUCCESS), "cusolverDnDpotrf failed");
    LOGGING_CHECK(!(sync_status != cudaSuccess), "cudaStreamSynchronize failed");
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
