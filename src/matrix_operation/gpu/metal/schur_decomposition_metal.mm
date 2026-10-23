#if LINALG_HAS_METAL
#include "include/common/linalg_macros.h"
#include "include/common/linalg_profiler.h"
#include "include/common/metal_handle.h"
#include "include/matrix_operation/schur_decomposition.h"
#include "src/matrix_operation/schur_decomposition_workspace.h"

namespace linalg
{
namespace
{
id<MTLComputePipelineState> schur_pipeline()
{
    static id<MTLComputePipelineState> pipeline = []
    {
        @autoreleasepool
        {
            constexpr char source[] = LINALG_METAL_CORE_PREAMBLE
#include "src/matrix_operation/schur_decomposition_impl.h"
R"(
kernel void schur_decomposition(device const float* a [[buffer(0)]],
    device float* sr [[buffer(1)]], device float* si [[buffer(2)]],
    device float* qr [[buffer(3)]], device float* qi [[buffer(4)]],
    device float* workspace [[buffer(5)]], constant int* dims [[buffer(6)]])
{
    if (!linalg_schur::decompose_real(a, dims[0], dims[1], sr, si, dims[2], qr, qi, dims[3],
        workspace, 1.1920928955078125e-7f))
        linalg_schur::poison_schur(dims[0], sr, si, dims[2], qr, qi, dims[3]);
}
)";
            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
            else
#endif
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> library = [detail::metal_device() newLibraryWithSource:@(source)
                options:options error:&error];
            LINALG_CHECK(library != nil, "schur_decomposition: Metal compilation failed: {}",
                error == nil ? "unknown error" : error.localizedDescription.UTF8String);
            id<MTLFunction> function = [library newFunctionWithName:@"schur_decomposition"];
            LINALG_CHECK(function != nil, "schur_decomposition: Metal kernel not found");
            auto state = [detail::metal_device() newComputePipelineStateWithFunction:function error:&error];
            LINALG_CHECK(state != nil, "schur_decomposition: Metal pipeline creation failed");
            return state;
        }
    }();
    return pipeline;
}
}  // namespace

template <>
void schur_decomposition<float, linalg_backend::METAL_BLAS>::apply(const float* a,
    linalg_int n, linalg_int lda, float* sr, float* si, linalg_int lds,
    float* qr, float* qi, linalg_int ldq, void* workspace, size_t bytes, void* stream)
{
    LINALG_FUNCTION();
    detail::check_schur_decomposition<float, linalg_backend::METAL_BLAS>(
        a, n, lda, sr, si, lds, qr, qi, ldq, workspace, bytes);
    @autoreleasepool
    {
        const auto wrap = [n](const float* p, linalg_int stride)
        {
            return detail::metal_buffer_from_pointer(const_cast<float*>(p),
                static_cast<size_t>((n - 1) * stride + n) * sizeof(float));
        };
        NSArray<id<MTLBuffer>>* buffers = @[wrap(a, lda), wrap(sr, lds), wrap(si, lds),
            wrap(qr, ldq), wrap(qi, ldq), detail::metal_buffer_from_pointer(workspace,
                detail::portable_schur_workspace_size<float>(n))];
        const int dims[] = {static_cast<int>(n), static_cast<int>(lda),
            static_cast<int>(lds), static_cast<int>(ldq)};
        id<MTLCommandBuffer> command = [detail::metal_queue_from_stream(stream) commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:schur_pipeline()];
        for (NSUInteger i = 0; i < buffers.count; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder setBytes:dims length:sizeof(dims) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        detail::metal_commit_and_wait(command);
    }
}

template <>
size_t schur_decomposition<float, linalg_backend::METAL_BLAS>::workspace_size(linalg_int n)
{
    return detail::portable_schur_workspace_size<float>(n);
}
}  // namespace linalg
#endif
