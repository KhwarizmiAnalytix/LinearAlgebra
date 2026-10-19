#if LINALG_HAS_BLAS && LINALG_HAS_MKL

#include "src/matrix_operation/cpu/blas/blas_runtime.h"
#if __has_include("isolated_blas_config.h")
#include "isolated_blas_config.h"
#endif
#ifndef LINALG_LAPACKE_RUNTIME_PATH
#define LINALG_LAPACKE_RUNTIME_PATH ""
#endif

#include <filesystem>
#include <gtest/gtest.h>
#include <mkl.h>
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

namespace
{
std::string provider_path(linalg::detail::blas_symbol function)
{
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<const char*>(function),
            &module))
    {
        return {};
    }
    char       path[32768];
    const auto size = GetModuleFileNameA(module, path, sizeof(path));
    return std::string(path, size);
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(function), &info))
    {
        return {};
    }
    return info.dli_fname;
#endif
}
}  // namespace

TEST(BackendIsolation, BlasSymbolsBelongToSelectedProvider)
{
    using linalg::detail::resolve_blas_symbol;
    const auto gemm = resolve_blas_symbol("cblas_dgemm");
    EXPECT_NE(gemm, reinterpret_cast<linalg::detail::blas_symbol>(&cblas_dgemm));
    ASSERT_FALSE(provider_path(gemm).empty());
    EXPECT_TRUE(std::filesystem::equivalent(provider_path(gemm), LINALG_BLAS_RUNTIME_PATH));

#if LINALG_HAS_LAPACKE
    const auto getrf = resolve_blas_symbol("LAPACKE_dgetrf");
    EXPECT_NE(getrf, reinterpret_cast<linalg::detail::blas_symbol>(&LAPACKE_dgetrf));
    ASSERT_FALSE(provider_path(getrf).empty());
    const char* lapacke_path =
        LINALG_LAPACKE_RUNTIME_PATH[0] ? LINALG_LAPACKE_RUNTIME_PATH : LINALG_BLAS_RUNTIME_PATH;
    EXPECT_TRUE(std::filesystem::equivalent(provider_path(getrf), lapacke_path));
#endif
}

TEST(BackendIsolation, MissingSymbolsFailWithoutFallingBackToMkl)
{
    EXPECT_THROW(
        linalg::detail::resolve_blas_symbol("LINALG_nonexistent_symbol"), std::runtime_error);
    EXPECT_THROW(linalg::detail::resolve_blas_symbol("MKL_Get_Version"), std::runtime_error);
}
#endif
