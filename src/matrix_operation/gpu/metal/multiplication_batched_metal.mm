#if LINALG_HAS_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "include/matrix_operation/multiplication_batched.h"
#include "include/common/metal_handle.h"
#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"

// batched_matrix_multiplication is a plain free function (not backend-
// templated, see include/matrix_operation/multiplication_batched.h and
// CMakeLists.txt's LINALG_ENABLE_CUBLAS/ROCBLAS/METAL mutual-exclusivity
// guard), so unlike the struct-template ops in this directory there is no
// linalg_backend tag to specialize against; only one GPU backend's
// definition of this symbol may be linked in at a time.
//
// METAL_BLAS defines only the float overload: Apple GPUs have no native
// double type (see include/common/backend.h), and this symbol -- unlike
// multiplication<T, METAL_BLAS> -- has no per-type specialization to simply
// omit, so the double overload is left undefined here; linking a caller
// that references batched_matrix_multiplication(double, ...) under
// LINALG_ENABLE_METAL fails at link time, the same way it would under a
// hypothetical CUDA_BLAS/ROCM_BLAS build missing that translation unit.
namespace linalg
{
namespace
{

// MPSMatrixDescriptor's batched initializer (rows/columns per matrix,
// `matrices` matrices stacked `matrixBytes` apart) matches this op's
// contract exactly: `count` contiguous dim x dim row-major matrices.
// MPSMatrixMultiplication batches automatically when every operand
// descriptor shares the same matrix count.
void launch_batched(linalg_int dim, linalg_int count, const float* a, const float* b, float* c,
    void* stream)
{
    LINALG_CHECK(a != nullptr && b != nullptr && c != nullptr,
        "batched_matrix_multiplication: a, b, and c must not be null");
    LINALG_CHECK(dim > 0 && count > 0,
        "batched_matrix_multiplication: dim and count must be positive");

    @autoreleasepool
    {
        const auto     d           = static_cast<NSUInteger>(dim);
        const auto     n           = static_cast<NSUInteger>(count);
        const NSUInteger row_bytes = d * sizeof(float);
        const NSUInteger mat_bytes = d * row_bytes;

        id<MTLDevice> device = detail::metal_device();

        id<MTLBuffer> buf_a = detail::metal_buffer_from_pointer(
            const_cast<float*>(a), static_cast<size_t>(mat_bytes) * static_cast<size_t>(n));
        id<MTLBuffer> buf_b = detail::metal_buffer_from_pointer(
            const_cast<float*>(b), static_cast<size_t>(mat_bytes) * static_cast<size_t>(n));
        id<MTLBuffer> buf_c = detail::metal_buffer_from_pointer(
            c, static_cast<size_t>(mat_bytes) * static_cast<size_t>(n));

        MPSMatrixDescriptor* desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:d
                                                    columns:d
                                                   matrices:n
                                                   rowBytes:row_bytes
                                                matrixBytes:mat_bytes
                                                   dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc];

        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                            transposeLeft:NO
                                                                           transposeRight:NO
                                                                               resultRows:d
                                                                            resultColumns:d
                                                                          interiorColumns:d
                                                                                    alpha:1.0
                                                                                     beta:0.0];

        id<MTLCommandQueue>  queue       = detail::metal_queue_from_stream(stream);
        id<MTLCommandBuffer> command_buf = [queue commandBuffer];
        [gemm encodeToCommandBuffer:command_buf leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        detail::metal_commit_and_wait(command_buf);
    }
}

}  // namespace

void batched_matrix_multiplication(
    linalg_int dim, linalg_int count, const float* a, const float* b, float* c, void* stream)
{
    launch_batched(dim, count, a, b, c, stream);
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
