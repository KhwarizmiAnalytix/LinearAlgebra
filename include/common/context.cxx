#include "include/common/context.h"

#include <cstdlib>
#include <cstring>

#include "include/common/configure.h"  // IWYU pragma: keep
#include "include/util/exception.h"

#ifdef LINALG_ENABLE_CUBLAS
#include <cuda_runtime_api.h>
#endif

namespace linalg
{
namespace
{

bool env_backend_to(const char* name, backend& out)
{
    if (std::strcmp(name, "scalar") == 0)
    {
        out = backend::scalar;
        return true;
    }
    if (std::strcmp(name, "blas") == 0 || std::strcmp(name, "blas_lapack") == 0)
    {
        out = backend::blas_lapack;
        return true;
    }
    if (std::strcmp(name, "mkl") == 0)
    {
        out = backend::mkl;
        return true;
    }
    if (std::strcmp(name, "cublas") == 0)
    {
        out = backend::cublas;
        return true;
    }
    return false;
}

}  // namespace

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

std::vector<backend> Context::cpu_preference_order() const
{
    if (has_override_)
    {
        return {override_backend_, backend::scalar};
    }

    std::vector<backend> order;
    if (has_mkl())
    {
        order.push_back(backend::mkl);
    }
    if (has_blas())
    {
        order.push_back(backend::blas_lapack);
    }
    order.push_back(backend::scalar);
    return order;
}

void Context::set_backend(backend b)
{
    switch (b)
    {
        case backend::scalar:
            break;
        case backend::mkl:
            if (!has_mkl())
            {
                LINALG_THROW("requested backend::mkl but LINALG_ENABLE_MKL was not compiled in");
            }
            break;
        case backend::blas_lapack:
            if (!has_blas())
            {
                LINALG_THROW(
                    "requested backend::blas_lapack but LINALG_ENABLE_BLAS was not compiled in");
            }
            break;
        case backend::cublas:
            if (!has_cuda())
            {
                LINALG_THROW(
                    "requested backend::cublas but no CUDA device is available "
                    "(or LINALG_ENABLE_CUBLAS was not compiled in)");
            }
            break;
    }
    override_backend_ = b;
    has_override_     = true;
}

void Context::clear_backend_override()
{
    has_override_ = false;
}

Context& globalContext()
{
    static Context* ctx = [] {
        auto* c = new Context();
        if (const char* env = std::getenv("LINALG_BACKEND"))
        {
            backend b{};
            if (env_backend_to(env, b))
            {
                // Best-effort: an unavailable backend named in the environment is
                // ignored rather than aborting process startup.
                switch (b)
                {
                    case backend::mkl:
                        if (c->has_mkl())
                        {
                            c->set_backend(b);
                        }
                        break;
                    case backend::blas_lapack:
                        if (c->has_blas())
                        {
                            c->set_backend(b);
                        }
                        break;
                    case backend::cublas:
                        if (c->has_cuda())
                        {
                            c->set_backend(b);
                        }
                        break;
                    case backend::scalar:
                        c->set_backend(b);
                        break;
                }
            }
        }
        return c;
    }();
    return *ctx;
}

}  // namespace linalg
