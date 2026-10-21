#if LINALG_HAS_METAL

#import <Metal/Metal.h>

#include "include/matrix_operation/transpose.h"
#include "include/common/metal_handle.h"
#include "include/common/metal_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// MPS has no standalone transpose primitive (MPSMatrixMultiplication only
// transposes an operand in the course of a product), so this is a small
// hand-written MSL kernel -- one thread per output element, each writing
// its own unique destination slot, same embarrassingly-parallel shape as
// multiplication_batched_metal.mm's dispatch. `src`/`dst` are distinct
// buffers (dst is caller-owned scratch); the result is blitted back over
// `m` in the same command buffer so apply() still transposes in place.
constexpr char kTransposeSource[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void linalg_transpose_f32(constant float* src [[buffer(0)]],
                                  device float* dst [[buffer(1)]],
                                  constant uint& rows [[buffer(2)]],
                                  constant uint& columns [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= columns || gid.y >= rows) {
        return;
    }
    dst[gid.x * rows + gid.y] = src[gid.y * columns + gid.x];
}
)";

id<MTLComputePipelineState> transpose_pipeline()
{
    static id<MTLComputePipelineState> pipeline = []
    {
        @autoreleasepool
        {
            NSError* error  = nil;
            id<MTLLibrary> library =
                [detail::metal_device() newLibraryWithSource:@(kTransposeSource)
                                                       options:nil
                                                         error:&error];
            LINALG_CHECK(library != nil, "metal: compiling transpose kernel failed");
            id<MTLFunction> function = [library newFunctionWithName:@"linalg_transpose_f32"];
            LINALG_CHECK(function != nil, "metal: transpose kernel function not found");
            id<MTLComputePipelineState> state =
                [detail::metal_device() newComputePipelineStateWithFunction:function error:&error];
            LINALG_CHECK(state != nil, "metal: building transpose pipeline state failed");
            return state;
        }
    }();
    return pipeline;
}

size_t transpose_workspace_size_impl(linalg_long rows, linalg_long columns)
{
    detail::workspace_arena arena;
    arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));
    return arena.total_size();
}

void transpose_impl(
    linalg_long rows, linalg_long columns, float* m, void* workspace, size_t workspace_size, void* stream)
{
    LINALG_CHECK(m != nullptr, "transpose: m must not be null");
    LINALG_CHECK(rows != 0 && columns != 0, "transpose: rows and columns must be positive");
    LINALG_CHECK(workspace_size >= transpose_workspace_size_impl(rows, columns),
        "transpose: workspace_size too small");
    detail::workspace_arena arena(workspace, workspace_size);
    float* scratch = arena.next<float>(static_cast<size_t>(rows) * static_cast<size_t>(columns));

    @autoreleasepool
    {
        const auto r = detail::checked_narrow<uint32_t>(rows);
        const auto c = detail::checked_narrow<uint32_t>(columns);
        const size_t bytes = sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(columns);

        id<MTLBuffer> buf_m       = detail::metal_buffer_from_pointer(m, bytes);
        id<MTLBuffer> buf_scratch = detail::metal_buffer_from_pointer(scratch, bytes);

        id<MTLCommandQueue>  queue       = detail::metal_queue_from_stream(stream);
        id<MTLCommandBuffer> command_buf = [queue commandBuffer];

        id<MTLComputeCommandEncoder> encoder = [command_buf computeCommandEncoder];
        [encoder setComputePipelineState:transpose_pipeline()];
        [encoder setBuffer:buf_m offset:0 atIndex:0];
        [encoder setBuffer:buf_scratch offset:0 atIndex:1];
        [encoder setBytes:&r length:sizeof(r) atIndex:2];
        [encoder setBytes:&c length:sizeof(c) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(c, r, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];

        id<MTLBlitCommandEncoder> blit = [command_buf blitCommandEncoder];
        [blit copyFromBuffer:buf_scratch sourceOffset:0 toBuffer:buf_m destinationOffset:0 size:bytes];
        [blit endEncoding];

        detail::metal_commit_and_wait(command_buf);
    }
}

}  // namespace

template <>
void transpose<float, linalg_backend::METAL_BLAS>::apply(linalg_long rows,
    linalg_long  columns,
    float*       m,
    void*        workspace,
    size_t       workspace_size,
    void*        stream)
{
    LINALG_SCOPE("transpose/metal_blas<float>");
    transpose_impl(rows, columns, m, workspace, workspace_size, stream);
}

template <>
size_t transpose<float, linalg_backend::METAL_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns)
{
    return transpose_workspace_size_impl(rows, columns);
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
