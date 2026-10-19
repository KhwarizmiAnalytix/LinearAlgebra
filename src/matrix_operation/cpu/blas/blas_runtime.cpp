#if LINALG_HAS_BLAS && LINALG_HAS_MKL

#include "blas_runtime.h"
#if __has_include("isolated_blas_config.h")
#include "isolated_blas_config.h"
#endif
#ifndef LINALG_BLAS_RUNTIME_PATH
#error                                                                                             \
    "Combined MKL/BLAS builds require LINALG_BLAS_RUNTIME_PATH (configured automatically by CMake)."
#endif
#ifndef LINALG_LAPACKE_RUNTIME_PATH
#define LINALG_LAPACKE_RUNTIME_PATH ""
#endif

#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace linalg::detail
{
namespace
{
#if defined(_WIN32)
using library_handle = HMODULE;
#else
using library_handle = void*;
#endif

library_handle open_provider(const char* path)
{
#if defined(_WIN32)
    // Search for the provider's dependencies alongside the DLL as well as in
    // the normal loader locations. Do not change the process DLL search path.
    const auto handle = LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!handle)
    {
        throw std::runtime_error(std::string("Cannot load BLAS provider ") + path +
                                 " (Windows error " + std::to_string(GetLastError()) + ")");
    }
#else
    int flags = RTLD_NOW | RTLD_LOCAL;
#if !defined(__APPLE__)
    // RTLD_LOCAL alone does not prevent the provider's internal LAPACK calls
    // from binding to already-global MKL symbols with a different integer ABI.
    flags |= RTLD_DEEPBIND;
#endif
    const auto handle = dlopen(path, flags);
    if (!handle)
    {
        throw std::runtime_error(
            std::string("Cannot load BLAS provider ") + path + ": " + dlerror());
    }
#endif
    // Handles intentionally live for the process lifetime, like linked vendor
    // libraries. Cached function pointers must survive static destruction.
    return handle;
}

blas_symbol lookup(library_handle handle, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<blas_symbol>(GetProcAddress(handle, name));
#else
    return reinterpret_cast<blas_symbol>(dlsym(handle, name));
#endif
}

library_handle checked_provider(const char* path)
{
    auto handle = open_provider(path);
    if (lookup(handle, "MKL_Get_Version"))
    {
        throw std::runtime_error(std::string("Independent BLAS provider resolves to MKL: ") + path);
    }
    return handle;
}
}  // namespace

blas_symbol resolve_blas_symbol(const char* name)
{
    static const auto blas   = checked_provider(LINALG_BLAS_RUNTIME_PATH);
    auto              handle = blas;
    const char*       path   = LINALG_BLAS_RUNTIME_PATH;
    if (std::strncmp(name, "LAPACKE_", 8) == 0 && LINALG_LAPACKE_RUNTIME_PATH[0] != '\0')
    {
        static const auto lapacke = checked_provider(LINALG_LAPACKE_RUNTIME_PATH);
        handle                    = lapacke;
        path                      = LINALG_LAPACKE_RUNTIME_PATH;
    }
    const auto function = lookup(handle, name);
    if (!function)
    {
        throw std::runtime_error(std::string("BLAS provider ") + path + " does not export " + name +
                                 ". Select matching CBLAS/LAPACKE shared libraries and headers.");
    }
    return function;
}
}  // namespace linalg::detail
#endif
