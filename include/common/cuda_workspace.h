/*
 * LinearAlgebra — GPU-facing alias for the shared workspace_arena.
 *
 * `linalg::detail::workspace_arena` has no CUDA dependency (see
 * include/common/workspace_arena.h) and is shared verbatim between the CPU
 * and GPU matrix_operation backends. This header is included by every
 * src/matrix_operation/cublas/ translation unit that needs workspace
 * arithmetic, gated behind LINALG_HAS_CUBLAS like every other GPU-only
 * header in this codebase.
 */
#pragma once

#if LINALG_HAS_CUBLAS

#include "include/common/workspace_arena.h"

#endif  // LINALG_HAS_CUBLAS
