#pragma once

#include <cstddef>  // for quarisma_long

#include "include/common/device.h"
#include "include/common/linear_algebra_export.h"  // for LINALG_API

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif


namespace linalg
{
// `device` selects host vs. CUDA device pointers (see include/common/device.h);
// CPU backend (scalar / blas_lapack / mkl) is chosen via linalg::globalContext().
// There is currently no cublas/cusolver backend registered for this op — the
// row-major/column-major and U/V-swap bookkeeping cuSOLVER's gesvd would need
// has not been validated against real hardware, so device_type::cuda throws
// rather than risk a silently-wrong GPU result; see svd_decomposition.cxx.
LINALG_API void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    float*      A,
    quarisma_long lda,
    float*      S,
    float*      U,
    quarisma_long ldu,
    float*      VT,
    quarisma_long ldv,
    device_type device = device_type::cpu);

LINALG_API void svd_decomposition(
    quarisma_long rows,
    quarisma_long columns,
    double*     A,
    quarisma_long lda,
    double*     S,
    double*     U,
    quarisma_long ldu,
    double*     VT,
    quarisma_long ldv,
    device_type device = device_type::cpu);

}  // namespace linalg