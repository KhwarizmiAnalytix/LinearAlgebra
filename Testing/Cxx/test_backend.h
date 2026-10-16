/*
 * LinearAlgebra — test-only "run against every backend this build has"
 * helper.
 *
 * The library itself never defaults to a backend: every linalg::*<T, Backend>
 * entry point requires the caller to name linalg_backend::native,
 * blas_lapacke, or mkl explicitly. Tests want end-to-end coverage of every
 * backend a given build actually compiled in, not just one of them, so
 * for_each_available_backend() calls its functor once per backend gated on
 * (test-side) the same LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
 * macros CMakeLists.txt defines. native is unconditional: the scalar
 * fallback always builds.
 */
#pragma once

#include <type_traits>

#include "include/common/backend.h"

namespace linalg_test
{

template <typename F> void for_each_available_backend(F&& f)
{
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::native>{});
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::blas_lapacke>{});
#endif
#if LINALG_HAS_MKL
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::mkl>{});
#endif
}

}  // namespace linalg_test
