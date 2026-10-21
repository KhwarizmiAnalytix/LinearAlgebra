#include "include/common/context.h"

#if LINALG_HAS_CUBLAS
#include <cuda_runtime_api.h>
#endif

#if LINALG_HAS_ROCBLAS
#include <hip/hip_runtime_api.h>
#endif

#if LINALG_HAS_MKL
#include <mkl.h>

#include "include/common/types.h"

// CMakeLists.txt pins MKL_INTERFACE to ilp64 so MKL_INT is 64-bit, matching
// the linalg_int/linalg_long parameters that MKL call sites (cholesky/LU/
// SVD/inversion/multiplication/transpose) pass straight through as MKL_INT*
// without narrowing. If a build ever picks up an lp64 MKL despite that pin
// (e.g. a stale MKLROOT), this fails to compile instead of silently
// truncating every size/stride at the MKL boundary.
static_assert(sizeof(MKL_INT) == sizeof(linalg::index_t),
    "MKL_INT must be 64-bit: LinearAlgebra pins MKL_INTERFACE=ilp64, so an "
    "lp64 MKL_INT indicates a build picked up the wrong MKL interface "
    "library.");
#endif

namespace linalg
{

bool Context::has_mkl() const noexcept
{
#if LINALG_HAS_MKL
    return true;
#else
    return false;
#endif
}

bool Context::has_blas() const noexcept
{
#if LINALG_HAS_BLAS
    return true;
#else
    return false;
#endif
}

bool Context::has_cuda() const noexcept
{
#if LINALG_HAS_CUBLAS
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

bool Context::has_rocm() const noexcept
{
#if LINALG_HAS_ROCBLAS
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#else
    return false;
#endif
}

Context& globalContext()
{
    static Context ctx;
    return ctx;
}

}  // namespace linalg
