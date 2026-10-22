#if LINALG_HAS_METAL

#include "include/common/linalg_profiler.h"
#include "include/common/metal_handle.h"
#include "include/matrix_operation/logarithm.h"
#include "include/matrix_operation/square_root.h"
#include "src/common/matrix_functions_workspace.h"

namespace linalg
{
namespace
{

// Compile the exact same numerical core as the CPU/CUDA/HIP implementations.
// The single device thread is a correctness-oriented portable implementation,
// not a tuned large-matrix kernel. All numerical storage remains caller-owned.
constexpr char kMatrixFunctionSource[] = "#include <metal_stdlib>\n"
                                         "using namespace metal;\n"
                                         "#define MF_INLINE inline\n"
                                         "#define MF_ADDRESS device\n"
#define LINALG_MATRIX_FUNCTIONS_CORE(...) #__VA_ARGS__
#include "src/common/matrix_functions_core.h"
                                         R"(
kernel void linalg_matrix_function(device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]], device float* workspace [[buffer(2)]],
    constant int* dimensions [[buffer(3)]])
{
    const int n = dimensions[0];
    const int lda = dimensions[1];
    const int ldc = dimensions[2];
    if (!linalg_matrix_functions::compute(a, n, lda, result, ldc, workspace,
            dimensions[3] != 0, 1.1920928955078125e-7f))
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result[i * ldc + j] = NAN;
            }
        }
    }
}
)";

id<MTLComputePipelineState> matrix_function_pipeline()
{
    static id<MTLComputePipelineState> pipeline = []
    {
        @autoreleasepool
        {
            NSError*           error   = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            // Domain checks and cancellation-sensitive complex arithmetic need
            // IEEE finite/NaN semantics, not Metal's default fast-math mode.
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *))
            {
                options.mathMode = MTLMathModeSafe;
            }
            else
#endif
            {
                // Required on pre-macOS-15 runtimes; the current SDK warns
                // even though this branch only serves that compatibility path.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> library =
                [detail::metal_device() newLibraryWithSource:@(kMatrixFunctionSource)
                                                     options:options
                                                       error:&error];
            LINALG_CHECK(library != nil,
                "metal: compiling matrix functions failed: {}",
                error == nil ? "unknown error" : error.localizedDescription.UTF8String);
            id<MTLFunction> function = [library newFunctionWithName:@"linalg_matrix_function"];
            LINALG_CHECK(function != nil, "metal: matrix function kernel not found");
            id<MTLComputePipelineState> state =
                [detail::metal_device() newComputePipelineStateWithFunction:function error:&error];
            LINALG_CHECK(state != nil, "metal: building matrix function pipeline failed");
            return state;
        }
    }();
    return pipeline;
}

void apply_matrix_function(const float* a,
    linalg_int                          n,
    linalg_int                          lda,
    float*                              result,
    linalg_int                          ldc,
    void*                               workspace,
    size_t                              workspace_size,
    void*                               stream,
    bool                                is_log)
{
    detail::check_matrix_function(a, n, lda, result, ldc, workspace, workspace_size);
    @autoreleasepool
    {
        const auto    input_elements  = static_cast<size_t>((n - 1) * lda + n);
        const auto    output_elements = static_cast<size_t>((n - 1) * ldc + n);
        id<MTLBuffer> input           = detail::metal_buffer_from_pointer(
            const_cast<float*>(a), input_elements * sizeof(float));
        id<MTLBuffer> output =
            detail::metal_buffer_from_pointer(result, output_elements * sizeof(float));
        id<MTLBuffer> scratch = detail::metal_buffer_from_pointer(
            workspace, detail::matrix_function_workspace_size<float>(n));
        const int dimensions[] = {
            static_cast<int>(n), static_cast<int>(lda), static_cast<int>(ldc), is_log ? 1 : 0};
        id<MTLCommandBuffer> command = [detail::metal_queue_from_stream(stream) commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:matrix_function_pipeline()];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBuffer:scratch offset:0 atIndex:2];
        [encoder setBytes:dimensions length:sizeof(dimensions) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        detail::metal_commit_and_wait(command);
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
    apply_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, true);
}

template <> size_t logarithm<float, linalg_backend::METAL_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<float>(n);
}

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
    apply_matrix_function(a, n, lda, result, ldc, workspace, workspace_size, stream, false);
}

template <> size_t square_root<float, linalg_backend::METAL_BLAS>::workspace_size(linalg_int n)
{
    return detail::matrix_function_workspace_size<float>(n);
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
