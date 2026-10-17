#pragma once

// LINALG_SCOPE(name) — zero-overhead profiling scope annotation.
//
// When Profiler::Profiler is linked (LINALG_HAS_PROFILER=1), each invocation
// records CPU wall-clock time for the duration of the enclosing scope and
// annotates it with `name` in the trace (chrome://tracing / Perfetto).  GPU
// activity (cuBLAS/cuSOLVER kernel timings) is captured automatically when a
// GPU-enabled profiler::session is active.
//
// When the Profiler is not present (LINALG_HAS_PROFILER=0), both macros
// expand to nothing and add zero compile-time and run-time cost.
//
// Usage inside a function body:
//   void foo::apply(...) {
//       LINALG_SCOPE("foo/native<float>");
//       ...
//   }
//
// LINALG_FUNCTION() is a variant that captures __FUNCTION__ automatically.

#if LINALG_HAS_PROFILER
#include "profiler.h"
#define LINALG_SCOPE(name) PROFILER_SCOPE(name)
#define LINALG_FUNCTION()  PROFILER_FUNCTION()
#else
#define LINALG_SCOPE(name) /* no-op: Profiler not linked */
#define LINALG_FUNCTION()  /* no-op: Profiler not linked */
#endif
