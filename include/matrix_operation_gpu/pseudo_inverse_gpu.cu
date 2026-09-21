#include "include/matrix_operation_gpu/pseudo_inverse_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <limits>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/svd_decomposition_gpu.h"
#include <include/logging.h>

// Combines U (rows x k row-major), S (length k), VT (k x columns
// row-major) into Ainv = V * diag(S+) * U^T (columns x rows row-major),
// one thread per output element — the direct device-side equivalent of the
// CPU pseudo_inverse's triple loop (include/matrix_operation/pseudo_inverse
// .cxx), favoring simple, obviously-correct per-element indexing over a
// cuBLAS dgmm/gemm chain that would need the same row-/column-major
// bookkeeping tricks the rest of this file uses for factorizations.
template <typename scalar_t>
static __global__ void pinv_combine_kernel(const scalar_t* U,
    const scalar_t*                                        S,
    const scalar_t*                                        VT,
    scalar_t                                                tol,
    int                                                      rows,
    int                                                      columns,
    int                                                      k,
    scalar_t*                                                Ainv)
{
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (c >= columns || r >= rows)
    {
        return;
    }
    scalar_t sum = scalar_t(0);
    for (int i = 0; i < k; ++i)
    {
        const scalar_t s   = S[i];
        const scalar_t inv = (s > tol) ? (scalar_t(1) / s) : scalar_t(0);
        sum += VT[i * columns + c] * inv * U[r * k + i];
    }
    Ainv[c * rows + r] = sum;
}

namespace linalg
{
namespace gpu
{
namespace
{

dim3 grid_2d(int width, int height, dim3 block)
{
    return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
        (static_cast<unsigned>(height) + block.y - 1) / block.y);
}

template <typename T, typename AmaxFn>
void pseudo_inverse_impl(AmaxFn amax,
    linalg_long              rows,
    linalg_long              columns,
    const T*                   A,
    linalg_long              lda,
    T*                         Ainv,
    linalg_long              ldai,
    T                          tol,
    cudaStream_t               stream)
{
    if (A == nullptr || Ainv == nullptr)
    {
        LOGGING_THROW("pseudo_inverse: A and Ainv must not be null");
    }
    if (rows == 0 || columns == 0)
    {
        LOGGING_THROW("pseudo_inverse: rows and columns must be positive");
    }
    const auto r = static_cast<int>(rows);
    const auto c = static_cast<int>(columns);
    if (static_cast<int>(lda) != c || static_cast<int>(ldai) != r)
    {
        LOGGING_THROW(
            "linalg::gpu::pseudo_inverse requires tightly packed lda/ldai (lda == columns, ldai == "
            "rows)");
    }

    const auto k = std::min(r, c);

    T*  S  = nullptr;
    T*  U  = nullptr;
    T*  VT = nullptr;
    int* svd_info = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&S), sizeof(T) * static_cast<size_t>(k)) != cudaSuccess)
    {
        LOGGING_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&U),
            sizeof(T) * static_cast<size_t>(r) * static_cast<size_t>(k)) != cudaSuccess)
    {
        cudaFree(S);
        LOGGING_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&VT),
            sizeof(T) * static_cast<size_t>(k) * static_cast<size_t>(c)) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        LOGGING_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&svd_info), sizeof(int)) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        LOGGING_THROW("cudaMalloc failed");
    }

    svd_decomposition(rows, columns, A, columns, S, U, static_cast<linalg_long>(k), VT, columns,
        svd_info, stream);

    int svd_status = -1;
    if (cudaMemcpyAsync(&svd_status, svd_info, sizeof(int), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        cudaFree(svd_info);
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    cudaFree(svd_info);
    if (svd_status != 0)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        LOGGING_THROW("linalg::gpu::svd_decomposition failed inside pseudo_inverse", svd_status);
    }

    auto cublas_handle = detail::cublas_handle_for_current_device();
    detail::set_stream(cublas_handle, stream);
    int idx_1based = 1;
    if (amax(cublas_handle, k, S, 1, &idx_1based) != CUBLAS_STATUS_SUCCESS)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        LOGGING_THROW("cublasI*amax failed");
    }

    T smax = T(0);
    if (cudaMemcpyAsync(&smax, S + (idx_1based - 1), sizeof(T), cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
    {
        cudaFree(S);
        cudaFree(U);
        cudaFree(VT);
        LOGGING_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    smax = std::fabs(smax);

    const T eff_tol =
        (tol >= T(0)) ? tol : std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(r, c)) * smax;

    const dim3 block(16, 16);
    pinv_combine_kernel<T>
        <<<grid_2d(c, r, block), block, 0, stream>>>(U, S, VT, eff_tol, r, c, k, Ainv);
    const auto launch_err = cudaGetLastError();

    // S/U/VT are still being read by the kernel just launched above; a bare
    // cudaFree here would only be safe by accident of the legacy default
    // stream's implicit whole-device sync (see cuda_handle.h's
    // synchronize_and_free doc comment) — explicitly wait for `stream`
    // first so this is correct for a caller-supplied non-default stream
    // too.
    cudaError_t sync_status = cudaSuccess;
    detail::synchronize_and_free(S, stream, &sync_status);
    cudaFree(U);
    cudaFree(VT);

    if (launch_err != cudaSuccess)
    {
        LOGGING_THROW("pinv_combine_kernel launch failed");
    }
    if (sync_status != cudaSuccess)
    {
        LOGGING_THROW("cudaStreamSynchronize failed");
    }
}

}  // namespace

void pseudo_inverse(linalg_long rows,
    linalg_long                 columns,
    const float*                  A,
    linalg_long                 lda,
    float*                        Ainv,
    linalg_long                 ldai,
    float                         tol,
    cudaStream_t                  stream)
{
    pseudo_inverse_impl(cublasIsamax, rows, columns, A, lda, Ainv, ldai, tol, stream);
}

void pseudo_inverse(linalg_long rows,
    linalg_long                 columns,
    const double*                 A,
    linalg_long                 lda,
    double*                       Ainv,
    linalg_long                 ldai,
    double                        tol,
    cudaStream_t                  stream)
{
    pseudo_inverse_impl(cublasIdamax, rows, columns, A, lda, Ainv, ldai, tol, stream);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
