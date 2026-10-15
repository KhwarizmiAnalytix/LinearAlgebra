#pragma once

#include <cstddef>
#include <cstdint>

#include "include/common/backend.h"
#include "include/common/linear_algebra_export.h"
#include "include/common/types.h"

namespace linalg
{

enum class matrix_norm_type : std::uint8_t
{
    FROBENIUS     = 1,
    ONE           = 2,  // max absolute column sum
    INFINITY_NORM = 3,  // max absolute row sum
    TWO           = 4,  // spectral norm: largest singular value (via svd_decomposition)
};

// Host-memory-only CPU entry point. Works on general rows x columns,
// row-major matrices (not just square). FROBENIUS/ONE/INFINITY_NORM are a
// direct reduction over A and do not depend on linalg::kActiveBackend. TWO
// routes through linalg::svd_backend, which is itself backend-dispatched
// (see svd_decomposition.h) — A is read-only here in every case. Because
// the op as a whole (the switch over `type`) has no backend axis of its own,
// only the `native` specialization is ever defined, and callers always name
// it explicitly rather than kActiveBackend. The template still exists, with
// the same explicit-member-specialization shape as cholesky_backend and
// friends, purely so this op's call site looks like every other op's.
//
// `workspace` must point to at least `matrix_norm_backend<T,B>::workspace_size(
// rows, columns, type)` bytes of caller-owned host memory. The size depends
// on `type`: only TWO needs scratch (the internal svd_decomposition call);
// FROBENIUS/ONE/INFINITY_NORM need none.
template <typename T, linalg_backend Backend> struct matrix_norm_backend
{
    LINALG_API static T apply(const T* A,
        linalg_long                    rows,
        linalg_long                    columns,
        linalg_long                    lda,
        matrix_norm_type               type,
        void*                          workspace,
        size_t                         workspace_size);

    LINALG_API static size_t workspace_size(
        linalg_long rows, linalg_long columns, matrix_norm_type type);
};

}  // namespace linalg
