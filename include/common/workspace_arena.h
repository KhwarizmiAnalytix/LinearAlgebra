/*
 * LinearAlgebra — caller-owned scratch, carved from one buffer.
 *
 * Per project policy, LinearAlgebra never allocates its own CPU/GPU memory
 * (allocation belongs to the caller, e.g. the separate KhwarizmiAnalytix/
 * Memory repo). Every matrix operation that needs scratch therefore exposes
 * a `*_workspace_size()` sibling the caller queries up front, then passes a
 * single caller-owned `(workspace, workspace_size)` buffer into the
 * operation itself, which slices it into typed sub-buffers with
 * `workspace_arena`. Used identically on the CPU side (src/matrix_operation/)
 * and the GPU side (src/matrix_operation_gpu/, where `workspace` is a
 * cudaMalloc'd device buffer instead of a host one) — the arena itself has
 * no device dependency, it only ever does pointer arithmetic over
 * `void*`/`size_t`.
 *
 * The same arena serves both the size query and the real carve so the two
 * can never drift apart: a "sizing" arena (constructed with no base pointer)
 * only accumulates `total_size()`, while a real arena carves actual
 * sub-pointers out of the caller's buffer. An operation's internal `_plan`
 * function calls `arena.next<T>(count)` in a fixed order either way, so
 * `*_workspace_size()` and the operation itself are guaranteed to agree on
 * layout.
 */
#pragma once

#include <cstddef>

#include <include/logging.h>

namespace linalg
{
namespace detail
{

class workspace_arena
{
public:
    // Sizing pass: no real buffer: next<T>() returns nullptr and only
    // total_size() accumulates.
    workspace_arena() : base_(nullptr), capacity_(0), sizing_(true) {}

    // Real carve: next<T>() returns pointers into `base`, bounds-checked
    // against `capacity`.
    workspace_arena(void* base, size_t capacity)
        : base_(static_cast<unsigned char*>(base)), capacity_(capacity), sizing_(false)
    {
    }

    // Default alignment is 64 bytes (a cache line, and enough for AVX-512)
    // rather than something larger: CPU ops routinely carve many small
    // buffers (a handful of `T`s each), so an unnecessarily large default
    // would pad every one of them for no benefit on host memory.
    template <typename T>
    T* next(size_t count, size_t alignment = 64)
    {
        const size_t aligned_offset = (offset_ + alignment - 1) / alignment * alignment;
        const size_t bytes          = sizeof(T) * count;
        offset_                     = aligned_offset + bytes;
        if (sizing_)
        {
            return nullptr;
        }
        LOGGING_CHECK(offset_ <= capacity_, "workspace_arena: workspace too small");
        return reinterpret_cast<T*>(base_ + aligned_offset);
    }

    size_t total_size() const { return offset_; }

private:
    unsigned char* base_;
    size_t         capacity_;
    size_t         offset_ = 0;
    bool           sizing_;
};

}  // namespace detail
}  // namespace linalg
