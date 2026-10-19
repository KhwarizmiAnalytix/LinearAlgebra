#pragma once

#if LINALG_HAS_MKL
#include "include/common/linear_algebra_export.h"

namespace linalg::detail
{
using blas_symbol = void (*)();
LINALG_API blas_symbol resolve_blas_symbol(const char* name);
}  // namespace linalg::detail

// Use the selected provider's declaration for the function-pointer type. Cache
// lookup on first use; subsequent operations only make an indirect call.
#define LINALG_BLAS_SYMBOL_NAME_IMPL(name) #name
#define LINALG_BLAS_SYMBOL_NAME(name) LINALG_BLAS_SYMBOL_NAME_IMPL(name)
#define LINALG_BLAS_FUNCTION(name)                                                                 \
    (                                                                                              \
        []                                                                                         \
        {                                                                                          \
            static const auto function = reinterpret_cast<decltype(&name)>(                        \
                ::linalg::detail::resolve_blas_symbol(LINALG_BLAS_SYMBOL_NAME(name)));             \
            return function;                                                                       \
        }())
#else
#define LINALG_BLAS_FUNCTION(name) (name)
#endif
