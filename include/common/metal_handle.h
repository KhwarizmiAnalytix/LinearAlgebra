/*
 * LinearAlgebra — shared Metal device/queue/buffer accessors.
 *
 * Counterpart to cuda_handle.h / hip_handle.h for the METAL_BLAS backend.
 * Apple Silicon and modern Intel Macs expose effectively one usable GPU per
 * process, so unlike the CUDA/HIP handle pools (keyed by the active device
 * index) this keeps a single process-wide MTLDevice/MTLCommandQueue pair,
 * created lazily on first use.
 *
 * Pointer convention: every METAL_BLAS entry point takes the same bare `T*`
 * (host-visible) pointers the NATIVE/BLAS_LAPACKE backends use, rather than
 * an opaque device handle, exactly as CUDA_BLAS takes a `cudaMalloc`'d
 * pointer and ROCM_BLAS a `hipMalloc`'d one. Metal has no equivalent of
 * "pass a raw device pointer into a kernel launch": a compute dispatch
 * needs an id<MTLBuffer>, so metal_buffer_from_pointer() below wraps the
 * caller's pointer into one with zero copies via
 * -[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]. That
 * API requires the wrapped pointer to be aligned to the OS page size — the
 * same "the caller provides a backend-appropriate pointer" contract
 * CUDA_BLAS/ROCM_BLAS already impose, just satisfied differently (e.g.
 * `posix_memalign`/`mmap` instead of `cudaMalloc`). This keeps
 * LinearAlgebra's "never allocates, including internal scratch" principle
 * intact (docs/backend_roadmap.md): a misaligned pointer is a caller error
 * reported via LINALG_CHECK, never silently papered over with a copy this
 * library would have to allocate.
 *
 * Every wrapped buffer uses MTLResourceStorageModeShared: on Apple Silicon's
 * unified memory architecture this is zero-copy and CPU/GPU-coherent once a
 * command buffer completes; on a discrete-GPU Intel Mac it still works, just
 * without the zero-copy benefit. apply() calls always commit and
 * `waitUntilCompleted` before returning (see multiplication_metal.mm etc.):
 * Metal has no caller-continuable stream handle the way a CUDA/HIP stream
 * is, so METAL_BLAS is synchronous regardless of the `stream` parameter,
 * which is only used to pick which id<MTLCommandQueue> to enqueue on.
 */
#pragma once

#if LINALG_HAS_METAL

#import <Metal/Metal.h>

#include <unistd.h>

#include <cstdint>

#include "include/common/linalg_checks.h"
#include "include/common/types.h"

namespace linalg
{
namespace detail
{

inline id<MTLDevice> metal_device()
{
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    LINALG_CHECK(device != nil, "MTLCreateSystemDefaultDevice failed: no Metal device available");
    return device;
}

inline id<MTLCommandQueue> metal_default_queue()
{
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    LINALG_CHECK(queue != nil, "MTLDevice newCommandQueue failed");
    return queue;
}

// `stream`, when non-null, is a caller-owned id<MTLCommandQueue> bridged
// through the same opaque `void*` every METAL_BLAS apply() takes (matching
// CUDA_BLAS's cudaStream_t / ROCM_BLAS's hipStream_t convention); null
// selects the process-wide default queue.
inline id<MTLCommandQueue> metal_queue_from_stream(void* stream)
{
    return stream == nullptr ? metal_default_queue()
                              : (__bridge id<MTLCommandQueue>)stream;
}

// Wraps a caller-owned raw pointer as a zero-copy MTLBuffer. `bytes` is
// rounded up to the OS page size for the API call (Metal's backing VM
// region for a page-aligned pointer already spans whole pages, and every
// METAL_BLAS kernel only ever touches the matrix's true logical extent, so
// the rounding never exposes out-of-bounds data) — see the file comment
// above for the alignment contract this relies on.
inline id<MTLBuffer> metal_buffer_from_pointer(void* ptr, size_t bytes)
{
    LINALG_CHECK(ptr != nullptr, "metal: pointer must not be null");
    LINALG_CHECK(bytes > 0, "metal: byte length must be positive");
    const size_t page = static_cast<size_t>(getpagesize());
    LINALG_CHECK(reinterpret_cast<std::uintptr_t>(ptr) % page == 0,
        "METAL_BLAS requires page-aligned buffers (allocate with posix_memalign/mmap "
        "at the system page size) -- see include/common/metal_handle.h");
    const size_t rounded = ((bytes + page - 1) / page) * page;
    id<MTLBuffer> buffer  = [metal_device() newBufferWithBytesNoCopy:ptr
                                                               length:rounded
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    LINALG_CHECK(
        buffer != nil, "MTLDevice newBufferWithBytesNoCopy failed (check pointer/length alignment)");
    return buffer;
}

// Commits `command_buffer`, waits for GPU completion, and throws if the
// work failed -- the single synchronization point every METAL_BLAS .mm file
// routes its dispatch through.
inline void metal_commit_and_wait(id<MTLCommandBuffer> command_buffer)
{
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    LINALG_CHECK(command_buffer.error == nil, "Metal command buffer execution failed");
}

}  // namespace detail
}  // namespace linalg

#endif  // LINALG_HAS_METAL
