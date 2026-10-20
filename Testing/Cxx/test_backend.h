/*
 * LinearAlgebra — test-only "run against every backend this build has"
 * helper.
 *
 * The library itself never defaults to a backend: every linalg::*<T, Backend>
 * entry point requires the caller to name linalg_backend::NATIVE,
 * BLAS_LAPACKE, MKL, or CUDA_BLAS explicitly. CPU tests cover every CPU
 * backend a given build actually compiled in, so
 * for_each_available_backend() calls its functor once per backend gated on
 * (test-side) the same LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
 * macros CMakeLists.txt defines. NATIVE is unconditional: the scalar
 * fallback always builds.
 *
 * CUDA_BLAS has a separate device-pointer API and cannot use these host
 * buffers. TestGpuMatrixOperations.cxx allocates device data and workspace
 * with cudaMalloc and runs when LINALG_ENABLE_CUBLAS is enabled.
 */
#pragma once

#include <type_traits>

#include "include/common/backend.h"

namespace linalg_test
{

template <typename F> void for_each_available_backend(F&& f)
{
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::NATIVE>{});
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::BLAS_LAPACKE>{});
#endif
#if LINALG_HAS_MKL
    f(std::integral_constant<linalg::linalg_backend, linalg::linalg_backend::MKL>{});
#endif
}

}  // namespace linalg_test
