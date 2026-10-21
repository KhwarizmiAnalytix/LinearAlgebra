#if LINALG_HAS_ROCBLAS

#include <hip/hip_runtime.h>

#include "include/matrix_operation/multiplication_batched.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

// One thread per output element, one z-block of the launch grid per matrix.
// C[row][col] = sum_k A[row][k] * B[k][col], all row-major. File-scope
// (not namespaced) __global__ template — only ever launched from this TU.
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
namespace
{

template <typename scalar_t>
void launch_batched(linalg_int dim,
    linalg_int         count,
    const scalar_t*      a,
    const scalar_t*      b,
    scalar_t*            c,
    hipStream_t          stream)
{
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        LINALG_THROW("batched_matrix_multiplication: a, b, and c must not be null");
    }
    if (dim <= 0 || count <= 0)
    {
        LINALG_THROW("batched_matrix_multiplication: dim and count must be positive");
    }
    const auto d = detail::checked_narrow<int>(dim);
    const auto n = detail::checked_narrow<int>(count);

    const dim3 block(16, 16);
    const dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y, n);

    batched_matmul_kernel<scalar_t><<<grid, block, 0, stream>>>(d, n, a, b, c);

    if (hipGetLastError() != hipSuccess)
    {
        LINALG_THROW("batched_matrix_multiplication kernel launch failed");
    }
}

}  // namespace

void batched_matrix_multiplication(linalg_int dim,
    linalg_int                                count,
    const float*                              a,
    const float*                              b,
    float*                                    c,
    void*                                     stream)
{
    launch_batched(dim, count, a, b, c, static_cast<hipStream_t>(stream));
}

void batched_matrix_multiplication(linalg_int dim,
    linalg_int                                count,
    const double*                             a,
    const double*                             b,
    double*                                   c,
    void*                                     stream)
{
    launch_batched(dim, count, a, b, c, static_cast<hipStream_t>(stream));
}

}  // namespace linalg

#endif  // LINALG_HAS_ROCBLAS
