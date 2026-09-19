#pragma once

#include <cstddef>

#include "include/common/device.h"
#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{

enum class cholesky_decomposition_enum : char
{
    LOWER_TRIANGULAR = 'L',
    UPPER_TRIANGULAR = 'U'
};

// `device` selects host vs. CUDA device pointers (see include/common/device.h);
// CPU backend (scalar / blas_lapack / mkl) is chosen via linalg::globalContext().
// cholesky_decomposition_aad (reverse-mode adjoint) has no vendor-library
// equivalent and always runs the scalar implementation.
LINALG_API bool cholesky_decomposition(
    float*       L,
    quarisma_int lda,
    linalg::cholesky_decomposition_enum type,
    device_type  device = device_type::cpu);

LINALG_API bool cholesky_decomposition_aad(
    float*                              L_aad,
    const float*                        L,
    quarisma_int                          lda,
    linalg::cholesky_decomposition_enum type,
    float*                              A_aad);

LINALG_API bool cholesky_decomposition(
    double*      L,
    quarisma_int lda,
    linalg::cholesky_decomposition_enum type,
    device_type  device = device_type::cpu);

LINALG_API bool cholesky_decomposition_aad(
    double*                             L_aad,
    const double*                       L,
    quarisma_int                          lda,
    linalg::cholesky_decomposition_enum type,
    double*                             A_aad);

}  // namespace linalg
