#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <vector>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation/lu_decomposition_dispatch.h"
#include "include/util/exception.h"

namespace linalg
{
namespace detail
{

// CAVEAT: cuSOLVER is column-major. The row-major buffer `m` (n x n, ld = n)
// is bit-identical in memory to its own transpose read column-major, so
// cusolverDnSgetrf/Dgetrf here factor A^T rather than A: it returns L', U',
// P such that P * A^T = L' * U' (column-major). Transposing back,
// A = P^T * U'^T * L'^T — an upper-times-lower ("UL") factorization in
// row-major, NOT the row-major "PA = LU" packed layout that LAPACKE_?getrf
// (and this library's own scalar/mkl/blas backends, and linear_solver's
// forward/backward substitution) assume. Callers that request
// device_type::cuda directly for the standalone factorization must consume
// the result accordingly; do not feed a device-computed factor into the
// CPU-side lu_solve/matrix_invert code paths, which assume the LAPACKE
// convention. Pivot values in `pivot` follow cuSOLVER's 1-based row-swap
// convention applied to A^T, i.e. to A's columns.
bool lu_cublas_f32(float* m, quarisma_int lda, quarisma_int* pivot)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();

    int lwork = 0;
    if (cusolverDnSgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSgetrf_bufferSize failed");
    }

    float* workspace = nullptr;
    int*   dev_ipiv  = nullptr;
    int*   dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(float) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_ipiv), sizeof(int) * static_cast<size_t>(n));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    const auto status = cusolverDnSgetrf(handle, n, n, m, n, workspace, dev_ipiv, dev_info);

    std::vector<int> host_ipiv(static_cast<size_t>(n));
    int              host_info = -1;
    cudaMemcpy(host_ipiv.data(), dev_ipiv, sizeof(int) * static_cast<size_t>(n), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_ipiv);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnSgetrf failed");
    }
    for (int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<quarisma_int>(host_ipiv[static_cast<size_t>(i)]);
    }
    return host_info == 0;
}

bool lu_cublas_f64(double* m, quarisma_int lda, quarisma_int* pivot)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = cusolver_handle();

    int lwork = 0;
    if (cusolverDnDgetrf_bufferSize(handle, n, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDgetrf_bufferSize failed");
    }

    double* workspace = nullptr;
    int*    dev_ipiv  = nullptr;
    int*    dev_info  = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(double) * static_cast<size_t>(lwork));
    cudaMalloc(reinterpret_cast<void**>(&dev_ipiv), sizeof(int) * static_cast<size_t>(n));
    cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

    const auto status = cusolverDnDgetrf(handle, n, n, m, n, workspace, dev_ipiv, dev_info);

    std::vector<int> host_ipiv(static_cast<size_t>(n));
    int              host_info = -1;
    cudaMemcpy(host_ipiv.data(), dev_ipiv, sizeof(int) * static_cast<size_t>(n), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(dev_ipiv);
    cudaFree(dev_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDnDgetrf failed");
    }
    for (int i = 0; i < n; ++i)
    {
        pivot[i] = static_cast<quarisma_int>(host_ipiv[static_cast<size_t>(i)]);
    }
    return host_info == 0;
}

LINALG_REGISTER_DISPATCH(lu_f32_stub, cuda, lu_cublas_f32);
LINALG_REGISTER_DISPATCH(lu_f64_stub, cuda, lu_cublas_f64);

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
