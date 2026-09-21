#include "include/matrix_operation_gpu/matrix_multiplication_batched_gpu.h"

#ifdef LINALG_ENABLE_CUBLAS

#include <cuda_runtime.h>

#include "include/util/exception.h"

// One thread per output element, one z-block of the launch grid per matrix.
// C[row][col] = sum_k A[row][k] * B[k][col], all row-major. File-scope
// (not namespaced) __global__ template, matching the usual CUDA idiom for a
// kernel that is only ever launched from this translation unit.
template <typename scalar_t>
static __global__ void batched_matmul_kernel(
    int dim, int count, const scalar_t* A_i, const scalar_t* B_i, scalar_t* C_i)
{
    const int matrix_id = blockIdx.z;
    if (matrix_id >= count)
    {
        return;
    }

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim || col >= dim)
    {
        return;
    }

    const int             matrix_size = dim * dim;
    const scalar_t* const A           = A_i + matrix_id * matrix_size;
    const scalar_t* const B           = B_i + matrix_id * matrix_size;
    scalar_t* const       C           = C_i + matrix_id * matrix_size;

    scalar_t sum = 0;
    for (int k = 0; k < dim; ++k)
    {
        sum += A[row * dim + k] * B[k * dim + col];
    }
    C[row * dim + col] = sum;
}

namespace linalg
{
namespace gpu
{
namespace
{

template <typename scalar_t>
void launch(quarisma_int dim, quarisma_int count, const scalar_t* a, const scalar_t* b, scalar_t* c)
{
    const auto d = static_cast<int>(dim);
    const auto n = static_cast<int>(count);

    const dim3 block(16, 16);
    const dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y, n);

    batched_matmul_kernel<scalar_t><<<grid, block>>>(d, n, a, b, c);

    if (cudaGetLastError() != cudaSuccess)
    {
        LINALG_THROW("batched_matrix_multiplication kernel launch failed");
    }
}

}  // namespace

void batched_matrix_multiplication(
    quarisma_int dim, quarisma_int count, const float* a, const float* b, float* c)
{
    launch(dim, count, a, b, c);
}

void batched_matrix_multiplication(
    quarisma_int dim, quarisma_int count, const double* a, const double* b, double* c)
{
    launch(dim, count, a, b, c);
}

}  // namespace gpu
}  // namespace linalg

#endif  // LINALG_ENABLE_CUBLAS
