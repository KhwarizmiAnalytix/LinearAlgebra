/*
 * LinearAlgebra — GPU-facing alias for the shared workspace_arena.
 *
 * `linalg::detail::workspace_arena` has no CUDA dependency (see
 * include/common/workspace_arena.h) and is shared verbatim between the CPU
 * and GPU matrix_operation backends; this header just keeps every existing
 * `#include "include/common/cuda_workspace.h"` in src/matrix_operation_gpu/
 * and include/matrix_operation_gpu/ working, gated the same way every other
 * GPU-only header in this codebase is.
 */
#pragma once

#ifdef LINALG_ENABLE_CUBLAS

#include "include/common/workspace_arena.h"

#endif  // LINALG_ENABLE_CUBLAS
