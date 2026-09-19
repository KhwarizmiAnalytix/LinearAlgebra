/*
 * LinearAlgebra — small compiler-portability macros used across
 * matrix_operation/. Reconstructed minimal stand-in for the (orphaned)
 * common/macros.h this code used to depend on: only the symbol(s) actually
 * referenced by matrix_operation call sites are provided.
 */
#pragma once

// Marks an intentionally-unused function parameter (e.g. `pivot` in the
// non-pivoting fallback paths of lu_decomposition / linear_solver /
// matrix_inversion). Placed as a leading attribute-specifier-seq on the
// parameter declaration, so plain [[maybe_unused]] is sufficient in C++17.
#define LINALG_UNUSED [[maybe_unused]]
