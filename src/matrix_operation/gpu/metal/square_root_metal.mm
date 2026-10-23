#if LINALG_HAS_METAL

#include "include/common/linalg_macros.h"
#include "include/common/linalg_profiler.h"
#include "include/common/metal_handle.h"
#include "include/matrix_operation/square_root.h"
#include "src/matrix_operation/square_root_workspace.h"

namespace linalg
{
namespace
{
inline id<MTLComputePipelineState> square_root_pipeline(unsigned stage)
{
    // Compile only the triangular and elementwise stages here. The complex
    // Schur reduction runs through schur_decomposition<METAL_BLAS> and dense
    // products go through multiplication<METAL_BLAS>, both with their own
    // existing Metal paths.
    static NSArray<id<MTLComputePipelineState>>* pipelines = []
    {
        @autoreleasepool
        {
            constexpr char source[] = LINALG_METAL_CORE_PREAMBLE
#include "src/common/square_root_core.h"
                R"(
kernel void square_root_schur(device const float* sr [[buffer(0)]],
    device const float* si [[buffer(1)]], device float* h [[buffer(2)]],
    device float* fr [[buffer(3)]], device float* fi [[buffer(4)]],
    constant int* dims [[buffer(5)]])
{
    linalg_schur::square_root_schur(sr, si, dims[0], h, fr, fi, 1.1920928955078125e-7f);
}
kernel void combine_square_root(device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]], device float* result [[buffer(2)]],
    constant int* dims [[buffer(3)]], constant float& sign [[buffer(4)]])
{
    linalg_schur::combine(a, b, result, dims[0], sign);
}
kernel void publish_square_root(device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]], device float* result [[buffer(2)]],
    constant int* dims [[buffer(3)]])
{
    if (!linalg_schur::publish(real, imag, dims[0], result,
        dims[1], 1.1920928955078125e-7f))
    {
        for (int i = 0; i < dims[0]; ++i)
        {
            for (int j = 0; j < dims[0]; ++j)
            {
                result[i * dims[1] + j] = NAN;
            }
        }
    }
}
)";
            NSError*           error   = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *))
            {
                options.mathMode = MTLMathModeSafe;
            }
            else
#endif
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> library = [detail::metal_device() newLibraryWithSource:@(source)
                                                                          options:options
                                                                            error:&error];
            LINALG_CHECK(library != nil,
                "metal: compiling square_root failed: {}",
                error == nil ? "unknown error" : error.localizedDescription.UTF8String);
            NSMutableArray<id<MTLComputePipelineState>>* states = [NSMutableArray new];
            for (NSString* name in
                @[ @"square_root_schur", @"combine_square_root", @"publish_square_root" ])
            {
                id<MTLFunction> function = [library newFunctionWithName:name];
                LINALG_CHECK(function != nil, "metal: square_root kernel not found");
                id<MTLComputePipelineState> state =
                    [detail::metal_device() newComputePipelineStateWithFunction:function
                                                                          error:&error];
                LINALG_CHECK(state != nil, "metal: building square_root pipeline failed");
                [states addObject:state];
            }
            return [states copy];
        }
    }();
    return pipelines[stage];
}

inline void dispatch_square_root(unsigned stage,
    NSArray<id<MTLBuffer>>*               buffers,
    const int*                            dimensions,
    size_t                                dimensions_bytes,
    void*                                 stream,
    const float*                          sign = nullptr)
{
    id<MTLCommandBuffer>         command = [detail::metal_queue_from_stream(stream) commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:square_root_pipeline(stage)];
    for (NSUInteger i = 0; i < buffers.count; ++i)
    {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [encoder setBytes:dimensions length:dimensions_bytes atIndex:buffers.count];
    if (sign != nullptr)
    {
        [encoder setBytes:sign length:sizeof(float) atIndex:buffers.count + 1];
    }
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    detail::metal_commit_and_wait(command);
}

void square_root_impl(const float* a,
    linalg_int                     n,
    linalg_int                     lda,
    float*                         result,
    linalg_int                     ldc,
    void*                          workspace,
    size_t                         workspace_size,
    void*                          stream)
{
    constexpr auto Backend = linalg_backend::METAL_BLAS;
    const auto     page    = static_cast<size_t>(getpagesize());
    detail::check_schur_function_arguments<float, Backend>(a, n, lda, result, ldc, workspace,
        workspace_size, detail::square_root_workspace_size<float, Backend>(n, page));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p = detail::square_root_plan<float, Backend>(arena, n, page);

    schur_decomposition<float, Backend>::apply(a, n, lda, p.schur.sr, p.schur.si, n, p.schur.qr,
        p.schur.qi, n, p.schur.schur_workspace, p.schur.schur_workspace_bytes, stream);

    @autoreleasepool
    {
        const auto nn   = static_cast<size_t>(n) * static_cast<size_t>(n);
        const auto wrap = [nn](const float* ptr)
        { return detail::metal_buffer_from_pointer(const_cast<float*>(ptr), nn * sizeof(float)); };
        id<MTLBuffer> h = detail::metal_buffer_from_pointer(p.h, 2 * nn * sizeof(float));
        const int     dims[] = {static_cast<int>(n)};
        dispatch_square_root(0,
            @[ wrap(p.schur.sr), wrap(p.schur.si), h, wrap(p.schur.fr), wrap(p.schur.fi) ],
            dims,
            sizeof(dims),
            stream);
        detail::reconstruct_schur_function<float, Backend>(p.schur,
            n,
            stream,
            [&](const float* x, const float* y, float* out, float sign)
            {
                const int count = static_cast<int>(nn);
                dispatch_square_root(
                    1, @[ wrap(x), wrap(y), wrap(out) ], &count, sizeof(count), stream, &sign);
            });
        id<MTLBuffer> output = detail::metal_buffer_from_pointer(
            result, static_cast<size_t>((n - 1) * ldc + n) * sizeof(float));
        const int publish_dims[] = {static_cast<int>(n), static_cast<int>(ldc)};
        dispatch_square_root(
            2, @[ wrap(p.schur.fr), wrap(p.schur.fi), output ], publish_dims, sizeof(publish_dims), stream);
    }
}

}  // namespace

template <>
void square_root<float, linalg_backend::METAL_BLAS>::apply(const float* a,
    linalg_int                                                          n,
    linalg_int                                                          lda,
    float*                                                              result,
    linalg_int                                                          ldc,
    void*                                                               workspace,
    size_t                                                              workspace_size,
    void*                                                               stream)
{
    LINALG_FUNCTION();
    square_root_impl(a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t square_root<float, linalg_backend::METAL_BLAS>::workspace_size(linalg_int n)
{
    return detail::square_root_workspace_size<float, linalg_backend::METAL_BLAS>(
        n, static_cast<size_t>(getpagesize()));
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
