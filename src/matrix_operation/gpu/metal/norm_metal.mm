#if LINALG_HAS_METAL

#import <Metal/Metal.h>

#include <cmath>

#include "include/matrix_operation/norm.h"
#include "include/common/metal_handle.h"
#include "include/common/metal_workspace.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// One thread per column/row, looping serially over the other dimension --
// same shape as norm_cublas.cu's abs_col_sum_kernel/abs_row_sum_kernel. The
// final max (ONE/INFINITY_NORM) or sum+sqrt (FROBENIUS) runs on the CPU
// over the small (rows- or columns-sized) partial-sum buffer: under Metal's
// MTLResourceStorageModeShared that buffer is the same unified memory on
// both sides, so once the command buffer completes (metal_commit_and_wait
// already waits) the raw workspace pointer is directly readable -- no
// device-to-host copy, and no second reduction kernel to get wrong.
constexpr char kNormSource[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void linalg_norm_abs_col_sum_f32(constant float* A [[buffer(0)]],
                                         device float* col_sums [[buffer(1)]],
                                         constant uint& rows [[buffer(2)]],
                                         constant uint& columns [[buffer(3)]],
                                         uint j [[thread_position_in_grid]])
{
    if (j >= columns) {
        return;
    }
    float sum = 0.0f;
    for (uint i = 0; i < rows; ++i) {
        sum += fabs(A[i * columns + j]);
    }
    col_sums[j] = sum;
}

kernel void linalg_norm_abs_row_sum_f32(constant float* A [[buffer(0)]],
                                         device float* row_sums [[buffer(1)]],
                                         constant uint& rows [[buffer(2)]],
                                         constant uint& columns [[buffer(3)]],
                                         uint i [[thread_position_in_grid]])
{
    if (i >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint j = 0; j < columns; ++j) {
        sum += fabs(A[i * columns + j]);
    }
    row_sums[i] = sum;
}

kernel void linalg_norm_sq_row_sum_f32(constant float* A [[buffer(0)]],
                                        device float* row_sums [[buffer(1)]],
                                        constant uint& rows [[buffer(2)]],
                                        constant uint& columns [[buffer(3)]],
                                        uint i [[thread_position_in_grid]])
{
    if (i >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint j = 0; j < columns; ++j) {
        const float v = A[i * columns + j];
        sum += v * v;
    }
    row_sums[i] = sum;
}
)";

struct norm_pipelines
{
    id<MTLComputePipelineState> abs_col_sum;
    id<MTLComputePipelineState> abs_row_sum;
    id<MTLComputePipelineState> sq_row_sum;
};

id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> library, NSString* name)
{
    NSError*        error    = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    LINALG_CHECK(function != nil, "metal: norm kernel function not found");
    id<MTLComputePipelineState> state =
        [detail::metal_device() newComputePipelineStateWithFunction:function error:&error];
    LINALG_CHECK(state != nil, "metal: building norm pipeline state failed");
    return state;
}

const norm_pipelines& pipelines()
{
    static norm_pipelines cached = []
    {
        @autoreleasepool
        {
            NSError*       error   = nil;
            id<MTLLibrary> library = [detail::metal_device() newLibraryWithSource:@(kNormSource)
                                                                            options:nil
                                                                              error:&error];
            LINALG_CHECK(library != nil, "metal: compiling norm kernels failed");
            norm_pipelines p;
            p.abs_col_sum = make_pipeline(library, @"linalg_norm_abs_col_sum_f32");
            p.abs_row_sum = make_pipeline(library, @"linalg_norm_abs_row_sum_f32");
            p.sq_row_sum  = make_pipeline(library, @"linalg_norm_sq_row_sum_f32");
            return p;
        }
    }();
    return cached;
}

size_t norm_workspace_size_impl(linalg_long rows, linalg_long columns, norm_type type)
{
    detail::workspace_arena arena;
    switch (type)
    {
    case norm_type::FROBENIUS:
    case norm_type::INFINITY_NORM:
        arena.next<float>(static_cast<size_t>(rows));
        break;
    case norm_type::ONE:
        arena.next<float>(static_cast<size_t>(columns));
        break;
    case norm_type::TWO:
    default:
        LINALG_THROW(
            "norm<metal_blas>: unsupported norm_type (TWO requires an on-device SVD, not "
            "available on METAL_BLAS)",
            static_cast<linalg_int>(type));
    }
    return arena.total_size();
}

float norm_impl(const float* A,
    linalg_long              rows,
    linalg_long              columns,
    linalg_long              lda,
    norm_type                type,
    void*                    workspace,
    size_t                   workspace_size,
    void*                    stream)
{
    LINALG_CHECK(A != nullptr, "norm: device input must not be null");
    LINALG_CHECK(rows > 0 && columns > 0, "norm: dimensions must be positive");
    LINALG_CHECK(lda == columns, "norm<metal_blas>: requires tightly packed lda");
    LINALG_CHECK(workspace_size >= norm_workspace_size_impl(rows, columns, type),
        "norm: workspace_size too small");

    const auto count = type == norm_type::ONE ? static_cast<size_t>(columns)
                                               : static_cast<size_t>(rows);
    detail::workspace_arena arena(workspace, workspace_size);
    float*                  partial = arena.next<float>(count);

    @autoreleasepool
    {
        const auto r = detail::checked_narrow<uint32_t>(rows);
        const auto c = detail::checked_narrow<uint32_t>(columns);

        id<MTLBuffer> buf_a =
            detail::metal_buffer_from_pointer(const_cast<float*>(A),
                sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(columns));
        id<MTLBuffer> buf_partial =
            detail::metal_buffer_from_pointer(partial, sizeof(float) * count);

        id<MTLComputePipelineState> pipeline = type == norm_type::ONE   ? pipelines().abs_col_sum
                                              : type == norm_type::INFINITY_NORM ? pipelines().abs_row_sum
                                                                                  : pipelines().sq_row_sum;

        id<MTLCommandQueue>  queue       = detail::metal_queue_from_stream(stream);
        id<MTLCommandBuffer> command_buf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_partial offset:0 atIndex:1];
        [encoder setBytes:&r length:sizeof(r) atIndex:2];
        [encoder setBytes:&c length:sizeof(c) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        detail::metal_commit_and_wait(command_buf);
    }

    if (type == norm_type::FROBENIUS)
    {
        float sum_of_squares = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            sum_of_squares += partial[i];
        }
        return std::sqrt(sum_of_squares);
    }

    float maximum = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        maximum = std::fmax(maximum, partial[i]);
    }
    return maximum;
}

}  // namespace

template <>
float norm<float, linalg_backend::METAL_BLAS>::apply(const float* A,
    linalg_long                                                   rows,
    linalg_long                                                   columns,
    linalg_long                                                   lda,
    norm_type                                                     type,
    void*                                                         workspace,
    size_t                                                        workspace_size,
    void*                                                         stream)
{
    LINALG_SCOPE("norm/metal_blas<float>");
    return norm_impl(A, rows, columns, lda, type, workspace, workspace_size, stream);
}

template <>
size_t norm<float, linalg_backend::METAL_BLAS>::workspace_size(
    linalg_long rows, linalg_long columns, norm_type type)
{
    return norm_workspace_size_impl(rows, columns, type);
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
