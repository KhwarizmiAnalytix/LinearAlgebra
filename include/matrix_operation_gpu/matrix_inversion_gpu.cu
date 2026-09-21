#include "include/matrix_operation_gpu/matrix_inversion_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "include/common/cuda_handle.h"
#include "include/matrix_operation_gpu/matrix_transpose_gpu.h"
#include "include/util/exception.h"

// File-scope (not namespaced) __global__ templates, matching the usual CUDA
// idiom already used by matrix_multiplication_batched_gpu.cu for a kernel
// only ever launched from its own translation unit.

// Identity is symmetric, so a flat row-major fill is correct whichever way
// the result is later read (row- or column-major).
template <typename scalar_t> static __global__ void fill_identity_kernel(int n, scalar_t* out)
{
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || col >= n)
    {
        return;
    }
    out[row * n + col] = (row == col) ? scalar_t(1) : scalar_t(0);
}

// Single-thread reduction: n is one matrix dimension, not a large array, so
// a parallel reduction buys nothing here. `m` is the row-major-packed L\U
// factor lu_decomposition_gpu-style code produces (see lu_decomposition_gpu
// .cxx) — the diagonal offset i*n+i is the same whether m is read row- or
// column-major, so this needs no transpose either way. `pivot` is
// cuSOLVER's own 1-based row-swap sequence for A (already verified to match
// the CPU lapacke_?getrf convention once `m` has been transposed before
// getrf — see lu_decomposition_gpu.cxx): each pivot[i] != i+1 entry flips
// the sign, matching lu_determinant's caller (matrix_determinant_helper) in
// include/matrix_operation/matrix_inversion.cxx's CPU counterpart.
template <typename scalar_t>
static __global__ void lu_determinant_kernel(
    const scalar_t* m, const int* pivot, int n, scalar_t* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    scalar_t det    = scalar_t(1);
    bool     negate = false;
    for (int i = 0; i < n; ++i)
    {
        det *= m[i * n + i];
        if (pivot[i] - 1 != i)
        {
            negate = !negate;
        }
    }
    *out = negate ? -det : det;
}

// Cholesky has no pivoting: det(A) = det(L)^2 = product(diag(L))^2.
template <typename scalar_t>
static __global__ void cholesky_determinant_kernel(const scalar_t* m, int n, scalar_t* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    scalar_t det = scalar_t(1);
    for (int i = 0; i < n; ++i)
    {
        det *= m[i * n + i];
    }
    *out = det * det;
}

namespace linalg
{
namespace gpu
{
namespace
{

dim3 square_grid(int n, dim3 block)
{
    return dim3((static_cast<unsigned>(n) + block.x - 1) / block.x,
        (static_cast<unsigned>(n) + block.y - 1) / block.y);
}

// LU-based in-place inversion: factor A (via the same transpose-getrf
// trick as lu_decomposition_gpu.cxx), solve A X = I for X = A^-1 against an
// on-device identity (nrhs = n), transpose the column-major solution back
// to row-major, and copy it over the caller's buffer in place — mirroring
// the CPU matrix_invert's factor-then-solve-against-identity strategy
// (include/matrix_operation/matrix_inversion.cxx) exactly, just with an
// explicit identity matrix standing in for LAPACKE's getri.
template <typename T, typename BufferSizeFn, typename GetrfFn, typename GetrsFn>
void lu_invert(BufferSizeFn buffer_size,
    GetrfFn                 getrf,
    GetrsFn                 getrs,
    T*                      m,
    quarisma_int            lda,
    int*                    info,
    cudaStream_t            stream)
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
    T*   identity  = nullptr;
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
    const auto nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (cudaMalloc(reinterpret_cast<void**>(&identity), sizeof(T) * nn) != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        LINALG_THROW("cudaMalloc failed");
    }

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        cudaFree(identity);
        LINALG_THROW("fill_identity_kernel launch failed");
    }

    // See lu_decomposition_gpu.cxx: transpose first so getrf factors A, not
    // A^T (matrix_transpose_gpu synchronizes `stream` internally before
    // returning, so no extra sync is needed around it here).
    matrix_transpose(lda, lda, m, stream);

    const auto status = getrf(handle, n, n, m, n, workspace, dev_ipiv, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        cudaFree(identity);
        LINALG_THROW("cusolverDn*getrf failed");
    }

    const auto solve_status = getrs(handle, CUBLAS_OP_N, n, n, m, n, dev_ipiv, identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        cudaFree(identity);
        LINALG_THROW("cusolverDn*getrs failed");
    }

    // `identity` now holds A^-1 column-major; transpose to row-major, then
    // copy over the caller's buffer in place.
    matrix_transpose(lda, lda, identity, stream);
    const auto copy_status =
        cudaMemcpyAsync(m, identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream);
    cudaError_t sync_status = cudaSuccess;
    if (copy_status == cudaSuccess)
    {
        sync_status = cudaStreamSynchronize(stream);
    }

    cudaFree(workspace);
    cudaFree(dev_ipiv);
    cudaFree(identity);

    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
}

// Cholesky-based in-place inversion: A is symmetric, so — unlike the LU
// path — no transpose is needed around potrf (see cholesky_decomposition_
// gpu.cxx); only the solution of A X = I still needs the transpose-back
// treatment, since X is not itself symmetric in general.
template <typename T, typename BufferSizeFn, typename PotrfFn, typename PotrsFn>
void cholesky_invert(BufferSizeFn buffer_size,
    PotrfFn                       potrf,
    PotrsFn                       potrs,
    T*                            m,
    quarisma_int                  lda,
    int*                          info,
    cudaStream_t                  stream)
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
    T* identity  = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    const auto nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (cudaMalloc(reinterpret_cast<void**>(&identity), sizeof(T) * nn) != cudaSuccess)
    {
        cudaFree(workspace);
        LINALG_THROW("cudaMalloc failed");
    }

    const dim3 block(16, 16);
    fill_identity_kernel<T><<<square_grid(n, block), block, 0, stream>>>(n, identity);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(identity);
        LINALG_THROW("fill_identity_kernel launch failed");
    }

    const auto status = potrf(handle, fill, n, m, n, workspace, lwork, info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(identity);
        LINALG_THROW("cusolverDn*potrf failed");
    }

    const auto solve_status = potrs(handle, fill, n, n, m, n, identity, n, info);
    if (solve_status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(identity);
        LINALG_THROW("cusolverDn*potrs failed");
    }

    matrix_transpose(lda, lda, identity, stream);
    const auto copy_status =
        cudaMemcpyAsync(m, identity, sizeof(T) * nn, cudaMemcpyDeviceToDevice, stream);
    cudaError_t sync_status = cudaSuccess;
    if (copy_status == cudaSuccess)
    {
        sync_status = cudaStreamSynchronize(stream);
    }

    cudaFree(workspace);
    cudaFree(identity);

    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
}

// `info` is internal scratch here (unlike matrix_invert, matrix_determinant
// exposes no info/pivot to the caller at all — the only observable output
// is the returned scalar): a structurally failed getrf/potrf call (bad
// arguments, workspace too small, ...) throws, but a numerically singular
// matrix (cuSOLVER's info > 0, factorization completed with a zero pivot)
// is not distinguished from a healthy call — its diagonal product is
// already ~0 in that case, so the returned determinant naturally matches
// the CPU matrix_determinant's silent-zero-on-singular behavior
// (matrix_determinant_helper in matrix_inversion.cxx) without extra
// bookkeeping.
template <typename T, typename BufferSizeFn, typename GetrfFn>
T lu_determinant(
    BufferSizeFn buffer_size, GetrfFn getrf, T* m, quarisma_int lda, cudaStream_t stream)
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
    int* dev_info  = nullptr;
    T*   dev_det   = nullptr;
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
    if (cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int)) != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&dev_det), sizeof(T)) != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        cudaFree(dev_info);
        LINALG_THROW("cudaMalloc failed");
    }

    matrix_transpose(lda, lda, m, stream);

    const auto status = getrf(handle, n, n, m, n, workspace, dev_ipiv, dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(dev_ipiv);
        cudaFree(dev_info);
        cudaFree(dev_det);
        LINALG_THROW("cusolverDn*getrf failed");
    }

    lu_determinant_kernel<T><<<1, 1, 0, stream>>>(m, dev_ipiv, n, dev_det);
    const auto launch_err = cudaGetLastError();

    T          det = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    cudaFree(workspace);
    cudaFree(dev_ipiv);
    cudaFree(dev_info);
    cudaFree(dev_det);

    if (launch_err != cudaSuccess)
    {
        LINALG_THROW("lu_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

template <typename T, typename BufferSizeFn, typename PotrfFn>
T cholesky_determinant(
    BufferSizeFn buffer_size, PotrfFn potrf, T* m, quarisma_int lda, cudaStream_t stream)
{
    const auto n      = static_cast<int>(lda);
    auto       handle = detail::cusolver_handle_for_current_device();
    detail::set_stream(handle, stream);
    const auto fill = CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    if (buffer_size(handle, fill, n, m, n, &lwork) != CUSOLVER_STATUS_SUCCESS)
    {
        LINALG_THROW("cusolverDn*potrf_bufferSize failed");
    }

    T*   workspace = nullptr;
    int* dev_info  = nullptr;
    T*   dev_det   = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&workspace), sizeof(T) * static_cast<size_t>(lwork)) !=
        cudaSuccess)
    {
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int)) != cudaSuccess)
    {
        cudaFree(workspace);
        LINALG_THROW("cudaMalloc failed");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&dev_det), sizeof(T)) != cudaSuccess)
    {
        cudaFree(workspace);
        cudaFree(dev_info);
        LINALG_THROW("cudaMalloc failed");
    }

    const auto status = potrf(handle, fill, n, m, n, workspace, lwork, dev_info);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        cudaFree(workspace);
        cudaFree(dev_info);
        cudaFree(dev_det);
        LINALG_THROW("cusolverDn*potrf failed");
    }

    cholesky_determinant_kernel<T><<<1, 1, 0, stream>>>(m, n, dev_det);
    const auto launch_err = cudaGetLastError();

    T          det = T(0);
    const auto copy_status =
        cudaMemcpyAsync(&det, dev_det, sizeof(T), cudaMemcpyDeviceToHost, stream);
    const auto sync_status = cudaStreamSynchronize(stream);

    cudaFree(workspace);
    cudaFree(dev_info);
    cudaFree(dev_det);

    if (launch_err != cudaSuccess)
    {
        LINALG_THROW("cholesky_determinant_kernel launch failed");
    }
    if (copy_status != cudaSuccess || sync_status != cudaSuccess)
    {
        LINALG_THROW("cudaMemcpyAsync/cudaStreamSynchronize failed");
    }
    return det;
}

}  // namespace

void matrix_invert(
    float* m, quarisma_int lda, int* info, linear_solver_type type, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_invert(
            cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, lda, info, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_invert(
            cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs, m, lda, info, stream);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

void matrix_invert(
    double* m, quarisma_int lda, int* info, linear_solver_type type, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        lu_invert(
            cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, lda, info, stream);
        break;
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        cholesky_invert(
            cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs, m, lda, info, stream);
        break;
    default:
        LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
    }
}

float matrix_determinant(float* m, quarisma_int lda, linear_solver_type type, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, m, lda, stream);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, m, lda, stream);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
}

double matrix_determinant(double* m, quarisma_int lda, linear_solver_type type, cudaStream_t stream)
{
    switch (type)
    {
    case linear_solver_type::LU_LINEAR_SOLVER:
    case linear_solver_type::LU_UPFRONT_LINEAR_SOLVER:
        return lu_determinant(cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, m, lda, stream);
    case linear_solver_type::CHOLESKY_LINEAR_SOLVER:
    case linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER:
        return cholesky_determinant(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, m, lda, stream);
    }
    LINALG_THROW("unsupported linear_solver_type", static_cast<quarisma_int>(type));
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
