#include "include/common/context.h"

#include "include/common/configure.h"  // IWYU pragma: keep

#ifdef LINALG_ENABLE_CUBLAS
#include <cuda_runtime_api.h>
#endif

namespace linalg
{

bool Context::has_mkl() const noexcept
{
#ifdef LINALG_ENABLE_MKL
    return true;
#else
    return false;
#endif
}

bool Context::has_blas() const noexcept
{
#ifdef LINALG_ENABLE_BLAS
    return true;
#else
    return false;
#endif
}

bool Context::has_cuda() const noexcept
{
#ifdef LINALG_ENABLE_CUBLAS
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
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
