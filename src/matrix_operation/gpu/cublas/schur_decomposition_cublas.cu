#if LINALG_HAS_CUBLAS
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

#include "include/common/linalg_profiler.h"
#include "include/matrix_operation/schur_decomposition.h"
#include "src/matrix_operation/schur_decomposition_impl.h"
#include "src/matrix_operation/schur_decomposition_workspace.h"

namespace linalg
{
namespace
{
template <typename T>
__global__ void schur_decomposition_kernel(const T* a,
    int                                             n,
    int                                             lda,
    T*                                              sr,
    T*                                              si,
    int                                             lds,
    T*                                              qr,
    T*                                              qi,
    int                                             ldq,
    T*                                              workspace,
    T                                               epsilon)
{
    if (!linalg_schur::decompose_real(a, n, lda, sr, si, lds, qr, qi, ldq, workspace, epsilon))
        linalg_schur::poison_schur(n, sr, si, lds, qr, qi, ldq);
}

template <typename T>
void apply_schur(const T* a,
    linalg_int            n,
    linalg_int            lda,
    T*                    sr,
    T*                    si,
    linalg_int            lds,
    T*                    qr,
    T*                    qi,
    linalg_int            ldq,
    void*                 workspace,
    size_t                bytes,
    void*                 stream)
{
    detail::check_schur_decomposition<T, linalg_backend::CUDA_BLAS>(
        a, n, lda, sr, si, lds, qr, qi, ldq, workspace, bytes);
    schur_decomposition_kernel<T><<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(a,
        static_cast<int>(n),
        static_cast<int>(lda),
        sr,
        si,
        static_cast<int>(lds),
        qr,
        qi,
        static_cast<int>(ldq),
        static_cast<T*>(workspace),
        std::numeric_limits<T>::epsilon());
    LINALG_CHECK(
        cudaGetLastError() == cudaSuccess, "schur_decomposition: device kernel launch failed");
}
}  // namespace

template <>
void schur_decomposition<float, linalg_backend::CUDA_BLAS>::apply(const float* a,
    linalg_int                                                                 n,
    linalg_int                                                                 lda,
    float*                                                                     sr,
    float*                                                                     si,
    linalg_int                                                                 lds,
    float*                                                                     qr,
    float*                                                                     qi,
    linalg_int                                                                 ldq,
    void*                                                                      workspace,
    size_t                                                                     bytes,
    void*                                                                      stream)
{
    LINALG_FUNCTION();
    apply_schur(a, n, lda, sr, si, lds, qr, qi, ldq, workspace, bytes, stream);
}

template <>
size_t schur_decomposition<float, linalg_backend::CUDA_BLAS>::workspace_size(linalg_int n)
{
    return detail::portable_schur_workspace_size<float>(n);
}

template <>
void schur_decomposition<double, linalg_backend::CUDA_BLAS>::apply(const double* a,
    linalg_int                                                                   n,
    linalg_int                                                                   lda,
    double*                                                                      sr,
    double*                                                                      si,
    linalg_int                                                                   lds,
    double*                                                                      qr,
    double*                                                                      qi,
    linalg_int                                                                   ldq,
    void*                                                                        workspace,
    size_t                                                                       bytes,
    void*                                                                        stream)
{
    LINALG_FUNCTION();
    apply_schur(a, n, lda, sr, si, lds, qr, qi, ldq, workspace, bytes, stream);
}

template <>
size_t schur_decomposition<double, linalg_backend::CUDA_BLAS>::workspace_size(linalg_int n)
{
    return detail::portable_schur_workspace_size<double>(n);
}
}  // namespace linalg
#endif  // LINALG_HAS_CUBLAS
