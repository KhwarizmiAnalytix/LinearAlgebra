#if LINALG_HAS_METAL
#include "include/common/linalg_profiler.h"
#include "include/common/metal_handle.h"
#include "include/matrix_operation/triangular_solver.h"
#include "src/common/triangular_solver_checks.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace linalg
{
template <>
void triangular_solver<float, linalg_backend::METAL_BLAS>::apply(bool upper,
    bool                                                              transpose,
    bool                                                              unit,
    linalg_int                                                        n,
    linalg_int                                                        nrhs,
    const float*                                                      a,
    linalg_int                                                        lda,
    const float*                                                      b,
    linalg_int                                                        ldb,
    float*                                                            x,
    linalg_int                                                        ldx,
    void* /*workspace*/,
    size_t /*workspace_size*/,
    void* stream)
{
    LINALG_FUNCTION();
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    @autoreleasepool
    {
        id<MTLBuffer> ba = detail::metal_buffer_from_pointer(const_cast<float*>(a),
            static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(float));
        id<MTLBuffer> bb = detail::metal_buffer_from_pointer(const_cast<float*>(b),
            static_cast<size_t>(n) * static_cast<size_t>(ldb) * sizeof(float));
        id<MTLBuffer> bx = detail::metal_buffer_from_pointer(
            x, static_cast<size_t>(n) * static_cast<size_t>(ldx) * sizeof(float));
        MPSMatrixDescriptor*      da = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                             columns:static_cast<NSUInteger>(n)
                            rowBytes:static_cast<NSUInteger>(lda) * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor*      db = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                             columns:static_cast<NSUInteger>(nrhs)
                            rowBytes:static_cast<NSUInteger>(ldb) * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor*      dx = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                             columns:static_cast<NSUInteger>(nrhs)
                            rowBytes:static_cast<NSUInteger>(ldx) * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix*                ma = [[MPSMatrix alloc] initWithBuffer:ba descriptor:da];
        MPSMatrix*                mb = [[MPSMatrix alloc] initWithBuffer:bb descriptor:db];
        MPSMatrix*                mx = [[MPSMatrix alloc] initWithBuffer:bx descriptor:dx];
        MPSMatrixSolveTriangular* solve =
            [[MPSMatrixSolveTriangular alloc] initWithDevice:detail::metal_device()
                                                       right:NO
                                                       upper:upper
                                                   transpose:transpose
                                                        unit:unit
                                                       order:static_cast<NSUInteger>(n)
                                      numberOfRightHandSides:static_cast<NSUInteger>(nrhs)
                                                       alpha:1.0];
        id<MTLCommandBuffer> command = [detail::metal_queue_from_stream(stream) commandBuffer];
        [solve encodeToCommandBuffer:command
                        sourceMatrix:ma
                 rightHandSideMatrix:mb
                      solutionMatrix:mx];
        detail::metal_commit_and_wait(command);
    }
}

template <>
size_t triangular_solver<float, linalg_backend::METAL_BLAS>::workspace_size(
    linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    return 0;
}
}  // namespace linalg
#endif
