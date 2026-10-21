/*
 * LinearAlgebra — GPU-facing alias for the shared workspace_arena.
 *
 * `linalg::detail::workspace_arena` has no Metal dependency (see
 * include/common/workspace_arena.h) and is shared verbatim between the CPU
 * and GPU matrix_operation backends. This header is included by every
 * src/matrix_operation/gpu/metal/ translation unit that needs workspace
 * arithmetic, gated behind LINALG_HAS_METAL like every other GPU-only
 * header in this codebase. Direct port of cuda_workspace.h / hip_workspace.h.
 */
#pragma once

#if LINALG_HAS_METAL

#include "include/common/workspace_arena.h"

#endif  // LINALG_HAS_METAL
