#if LINALG_HAS_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "include/matrix_operation/multiplication.h"
#include "include/common/metal_handle.h"
#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/narrow.h"

namespace linalg
{
namespace
{

// Unlike cuBLAS/hipBLAS (column-major, requiring the operand-swap identity
// used by multiplication_cublas.cpp/multiplication_hip.cpp), MPSMatrix is
// natively row-major (rowBytes is the stride between consecutive rows), so
// a/b/c map onto MPSMatrixDescriptor directly with no relabeling trick.
void multiplication_impl(bool transpose_a,
    bool        transpose_b,
    linalg_int  rows,
    linalg_int  columns,
    linalg_int  depth,
    const float* a,
    linalg_int  lda,
    const float* b,
    linalg_int  ldb,
    float*      c,
    linalg_int  ldc,
    void*       stream)
{
    LINALG_CHECK(a != nullptr && b != nullptr && c != nullptr,
        "multiplication: a, b, and c must not be null");
    LINALG_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0,
        "multiplication: rows/columns/depth/lda/ldb/ldc must be positive");

    @autoreleasepool
    {
        const auto a_rows = static_cast<NSUInteger>(transpose_a ? depth : rows);
        const auto a_cols = static_cast<NSUInteger>(transpose_a ? rows : depth);
        const auto b_rows = static_cast<NSUInteger>(transpose_b ? columns : depth);
        const auto b_cols = static_cast<NSUInteger>(transpose_b ? depth : columns);

        id<MTLDevice> device = detail::metal_device();

        id<MTLBuffer> buf_a = detail::metal_buffer_from_pointer(
            const_cast<float*>(a), sizeof(float) * static_cast<size_t>(lda) * a_rows);
        id<MTLBuffer> buf_b = detail::metal_buffer_from_pointer(
            const_cast<float*>(b), sizeof(float) * static_cast<size_t>(ldb) * b_rows);
        id<MTLBuffer> buf_c = detail::metal_buffer_from_pointer(
            c, sizeof(float) * static_cast<size_t>(ldc) * static_cast<size_t>(rows));

        MPSMatrixDescriptor* desc_a =
            [MPSMatrixDescriptor matrixDescriptorWithRows:a_rows
                                                    columns:a_cols
                                                   rowBytes:static_cast<NSUInteger>(lda) * sizeof(float)
                                                   dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b =
            [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows
                                                    columns:b_cols
                                                   rowBytes:static_cast<NSUInteger>(ldb) * sizeof(float)
                                                   dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c =
            [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
                                                    columns:static_cast<NSUInteger>(columns)
                                                   rowBytes:static_cast<NSUInteger>(ldc) * sizeof(float)
                                                   dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

        MPSMatrixMultiplication* gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:transpose_a
                                             transposeRight:transpose_b
                                                 resultRows:static_cast<NSUInteger>(rows)
                                              resultColumns:static_cast<NSUInteger>(columns)
                                            interiorColumns:static_cast<NSUInteger>(depth)
                                                      alpha:1.0
                                                       beta:0.0];

        id<MTLCommandQueue>  queue        = detail::metal_queue_from_stream(stream);
        id<MTLCommandBuffer> command_buf  = [queue commandBuffer];
        [gemm encodeToCommandBuffer:command_buf leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        detail::metal_commit_and_wait(command_buf);
    }
}

}  // namespace

template <>
void multiplication<float, linalg_backend::METAL_BLAS>::apply(bool transpose_a,
    bool           transpose_b,
    linalg_int     rows,
    linalg_int     columns,
    linalg_int     depth,
    const float*   a,
    linalg_int     lda,
    const float*   b,
    linalg_int     ldb,
    float*         c,
    linalg_int     ldc,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void*          stream)
{
    LINALG_SCOPE("multiplication/metal_blas<float>");
    multiplication_impl(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc, stream);
}

template <>
size_t multiplication<float, linalg_backend::METAL_BLAS>::workspace_size(
    linalg_int /*rows*/, linalg_int /*columns*/, linalg_int /*depth*/)
{
    return 0;
}

}  // namespace linalg

#endif  // LINALG_HAS_METAL
