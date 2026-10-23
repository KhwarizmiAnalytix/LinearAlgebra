#if LINALG_HAS_METAL

#include "include/common/linalg_macros.h"
#include "include/common/linalg_profiler.h"
#include "include/common/metal_handle.h"
#include "include/matrix_operation/logarithm.h"
#include "src/matrix_operation/logarithm_workspace.h"

namespace linalg
{
namespace
{
inline id<MTLComputePipelineState> logarithm_pipeline(unsigned stage)
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
#include "src/common/logarithm_core.h"
                R"(
kernel void prepare_logarithm(device const float* sr [[buffer(0)]],
    device const float* si [[buffer(1)]], device float* h [[buffer(2)]],
    device float* f [[buffer(3)]], device float* diagonal [[buffer(4)]],
    device float* state [[buffer(5)]], constant int* dims [[buffer(6)]])
{
    linalg_schur::logarithm_schur_prepare(sr, si, dims[0], h, f, diagonal, state,
        1.1920928955078125e-7f);
}
kernel void pack_logarithm_solve(device const float* h [[buffer(0)]],
    device const float* state [[buffer(1)]], device float* a [[buffer(2)]],
    device float* b [[buffer(3)]], constant int* dims [[buffer(4)]])
{
    linalg_schur::pack_logarithm_solve(h, state, a, b, dims[0], dims[1]);
}
kernel void accumulate_logarithm_solve(device float* f [[buffer(0)]],
    device const float* state [[buffer(1)]], device const float* x [[buffer(2)]],
    constant int* dims [[buffer(3)]])
{
    linalg_schur::accumulate_logarithm_solve(f, state, x, dims[0], dims[1]);
}
kernel void finish_logarithm(device float* f [[buffer(0)]],
    device const float* state [[buffer(1)]], device const float* diagonal [[buffer(2)]],
    device float* fr [[buffer(3)]], device float* fi [[buffer(4)]],
    constant int* dims [[buffer(5)]])
{
    linalg_schur::logarithm_schur_finish(f, state, diagonal, dims[0], fr, fi);
}
kernel void combine_logarithm(device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]], device float* result [[buffer(2)]],
    constant int* dims [[buffer(3)]], constant float& sign [[buffer(4)]])
{
    linalg_schur::combine(a, b, result, dims[0], sign);
}
kernel void publish_logarithm(device const float* real [[buffer(0)]],
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
                "metal: compiling logarithm failed: {}",
                error == nil ? "unknown error" : error.localizedDescription.UTF8String);
            NSMutableArray<id<MTLComputePipelineState>>* states = [NSMutableArray new];
            for (NSString* name in @[
                     @"prepare_logarithm",
                     @"combine_logarithm",
                     @"publish_logarithm",
                     @"pack_logarithm_solve",
                     @"accumulate_logarithm_solve",
                     @"finish_logarithm"
                 ])
            {
                id<MTLFunction> function = [library newFunctionWithName:name];
                LINALG_CHECK(function != nil, "metal: logarithm kernel not found");
                id<MTLComputePipelineState> state =
                    [detail::metal_device() newComputePipelineStateWithFunction:function
                                                                          error:&error];
                LINALG_CHECK(state != nil, "metal: building logarithm pipeline failed");
                [states addObject:state];
            }
            return [states copy];
        }
    }();
    return pipelines[stage];
}

inline void dispatch_logarithm(unsigned stage,
    NSArray<id<MTLBuffer>>*             buffers,
    const int*                          dimensions,
    size_t                              dimensions_bytes,
    void*                               stream,
    const float*                        sign = nullptr)
{
    id<MTLCommandBuffer>         command = [detail::metal_queue_from_stream(stream) commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:logarithm_pipeline(stage)];
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

void logarithm_impl(const float* a,
    linalg_int                   n,
    linalg_int                   lda,
    float*                       result,
    linalg_int                   ldc,
    void*                        workspace,
    size_t                       workspace_size,
    void*                        stream)
{
    constexpr auto Backend = linalg_backend::METAL_BLAS;
    const auto     page    = static_cast<size_t>(getpagesize());
    detail::check_logarithm_order(n);
    detail::check_schur_function_arguments<float, Backend>(a, n, lda, result, ldc, workspace,
        workspace_size, detail::logarithm_workspace_size<float, Backend>(n, page));
    detail::workspace_arena arena(workspace, workspace_size);
    const auto              p = detail::logarithm_plan<float, Backend>(arena, n, page);

    schur_decomposition<float, Backend>::apply(a, n, lda, p.schur.sr, p.schur.si, n, p.schur.qr,
        p.schur.qi, n, p.schur.schur_workspace, p.schur.schur_workspace_bytes, stream);

    @autoreleasepool
    {
        const auto nn   = static_cast<size_t>(n) * static_cast<size_t>(n);
        const auto wrap = [nn](const float* ptr)
        { return detail::metal_buffer_from_pointer(const_cast<float*>(ptr), nn * sizeof(float)); };
        id<MTLBuffer> h        = detail::metal_buffer_from_pointer(p.h, 2 * nn * sizeof(float));
        id<MTLBuffer> f        = detail::metal_buffer_from_pointer(p.f, 2 * nn * sizeof(float));
        id<MTLBuffer> diagonal = detail::metal_buffer_from_pointer(
            p.diagonal, 2 * static_cast<size_t>(n) * sizeof(float));
        id<MTLBuffer> state  = detail::metal_buffer_from_pointer(p.state, 3 * sizeof(float));
        id<MTLBuffer> solve_a = detail::metal_buffer_from_pointer(p.triangular_a, 4 * nn * sizeof(float));
        id<MTLBuffer> solve_b = detail::metal_buffer_from_pointer(p.triangular_b, 2 * nn * sizeof(float));
        id<MTLBuffer> solve_x = detail::metal_buffer_from_pointer(p.triangular_x, 2 * nn * sizeof(float));
        const int prepare_dims[] = {static_cast<int>(n)};
        dispatch_logarithm(0,
            @[ wrap(p.schur.sr), wrap(p.schur.si), h, f, diagonal, state ],
            prepare_dims,
            sizeof(prepare_dims),
            stream);
        for (int node = 0; node < 7; ++node)
        {
            const int dims[] = {static_cast<int>(n), node};
            dispatch_logarithm(3, @[ h, state, solve_a, solve_b ], dims, sizeof(dims), stream);
            triangular_solver<float, Backend>::apply(true,
                false,
                true,
                2 * n,
                n,
                p.triangular_a,
                2 * n,
                p.triangular_b,
                n,
                p.triangular_x,
                n,
                p.triangular_workspace,
                p.triangular_bytes,
                stream);
            dispatch_logarithm(4, @[ f, state, solve_x ], dims, sizeof(dims), stream);
        }
        const int finish_dims[] = {static_cast<int>(n)};
        dispatch_logarithm(5,
            @[ f, state, diagonal, wrap(p.schur.fr), wrap(p.schur.fi) ],
            finish_dims,
            sizeof(finish_dims),
            stream);
        detail::reconstruct_schur_function<float, Backend>(p.schur,
            n,
            stream,
            [&](const float* x, const float* y, float* out, float sign)
            {
                const int count = static_cast<int>(nn);
                dispatch_logarithm(
                    1, @[ wrap(x), wrap(y), wrap(out) ], &count, sizeof(count), stream, &sign);
            });
        id<MTLBuffer> output = detail::metal_buffer_from_pointer(
            result, static_cast<size_t>((n - 1) * ldc + n) * sizeof(float));
        const int publish_dims[] = {static_cast<int>(n), static_cast<int>(ldc)};
        dispatch_logarithm(
            2, @[ wrap(p.schur.fr), wrap(p.schur.fi), output ], publish_dims, sizeof(publish_dims), stream);
    }
}

}  // namespace

template <>
void logarithm<float, linalg_backend::METAL_BLAS>::apply(const float* a,
    linalg_int                                                        n,
    linalg_int                                                        lda,
    float*                                                            result,
    linalg_int                                                        ldc,
    void*                                                             workspace,
    size_t                                                            workspace_size,
    void*                                                             stream)
{
    LINALG_FUNCTION();
    logarithm_impl(a, n, lda, result, ldc, workspace, workspace_size, stream);
}

template <> size_t logarithm<float, linalg_backend::METAL_BLAS>::workspace_size(linalg_int n)
{
    detail::check_logarithm_order(n);
    return detail::logarithm_workspace_size<float, linalg_backend::METAL_BLAS>(
        n, static_cast<size_t>(getpagesize()));
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
