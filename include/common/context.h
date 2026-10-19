/*
 * LinearAlgebra — compiled-in backend introspection.
 *
 * Which CPU/GPU backend a matrix_operation call actually runs is a
 * compile-time choice (see e.g. multiplication.cpp: #if
 * LINALG_HAS_MKL / #elif LINALG_HAS_BLAS / #else scalar) — there is no
 * runtime dispatch to steer, unlike PyTorch's at::Context, which this used
 * to mirror more closely. What remains here is read-only introspection of
 * what a given build actually has available, for callers that want to know
 * before calling into an op (e.g. to decide whether the
 * linalg_backend::cuda_blas struct specializations are usable at all).
 */
#pragma once

#include "include/common/linear_algebra_export.h"

namespace linalg
{

class LINALG_API Context
{
public:
    // Compiled-in AND usable right now (for cuda: a device is actually present).
    bool has_mkl() const noexcept;
    bool has_blas() const noexcept;
    bool has_cuda() const noexcept;
};

LINALG_API Context& globalContext();

}  // namespace linalg
