#pragma once

#include <cstddef>

#include "include/common/linear_algebra_export.h"

#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define quarisma_int __int64
#define quarisma_long unsigned __int64
#else
#define quarisma_int long long int
#define quarisma_long unsigned long long int
#endif

namespace linalg
{
namespace gpu
{

// `m`, `pivot`, `info` are CUDA device pointers; only available when built
// with LINALG_ENABLE_CUBLAS. `pivot` must point to at least `lda` ints of
// device memory and `info` to one; neither is copied to or from host memory
// by this call, so it never blocks on a device synchronization — inspect
// them explicitly (e.g. cudaMemcpyAsync on your own stream) only when/if you
// need the result.
//
// CAVEAT: cuSOLVER is column-major. The row-major buffer `m` (n x n, ld = n)
// is bit-identical in memory to its own transpose read column-major, so
// cusolverDnSgetrf/Dgetrf here factor A^T rather than A: it returns L', U',
// P such that P * A^T = L' * U' (column-major). Transposing back,
// A = P^T * U'^T * L'^T — an upper-times-lower ("UL") factorization in
// row-major, NOT the row-major "PA = LU" packed layout that the CPU
// lu_decomposition (and linear_solver's forward/backward substitution)
// assumes. Do not feed a GPU-computed factor into the CPU-side
// lu_solve/matrix_invert code paths. Pivot values in `pivot` follow
// cuSOLVER's 1-based row-swap convention applied to A^T, i.e. to A's
// columns.
LINALG_API void lu_decomposition(float* m, quarisma_int lda, int* pivot, int* info);

LINALG_API void lu_decomposition(double* m, quarisma_int lda, int* pivot, int* info);

}  // namespace gpu
}  // namespace linalg
