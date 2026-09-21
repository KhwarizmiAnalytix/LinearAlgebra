#include "include/matrix_operation_gpu/lu_decomposition_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/util/exception.h"

namespace linalg
{
namespace gpu
{

void lu_decomposition(float* m, quarisma_int lda, int* pivot, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();

    int lwork = 0;
    if (cusolverDnSgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSgetrf_bufferSize failed");
    }

    float* workspace = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(float) * static_cast<size_t>(lwork));

    if (cusolverDnSgetrf(handle, n, n, m, n, workspace, pivot, info) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        LINALG_THROW("cusolverDnSgetrf failed");
    }
    cudaFree(workspace);
}

void lu_decomposition(double* m, quarisma_int lda, int* pivot, int* info)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle();

    int lwork = 0;
    if (cusolverDnDgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDgetrf_bufferSize failed");
    }

    double* workspace = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(double) * static_cast<size_t>(lwork));

    if (cusolverDnDgetrf(handle, n, n, m, n, workspace, pivot, info) != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        LINALG_THROW("cusolverDnDgetrf failed");
    }
    cudaFree(workspace);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
