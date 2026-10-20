#include "include/common/linalg_checks.h"
#include "include/common/linalg_profiler.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/qr_decomposition.h"
#include <algorithm>
#include <cmath>
#if LINALG_HAS_VECTORIZATION
#include "src/common/packet.h"
#endif

namespace linalg
{
namespace
{
template <typename T> inline void qr_axpy(T* y, const T* x, T alpha, linalg_long len)
{
#if LINALG_HAS_VECTORIZATION
    using P                     = packet<T>;
    constexpr linalg_long width = P::size;

    typename P::packet_t pa;
    P::set(alpha, pa);

    linalg_long c = 0;
    for (; c + width <= len; c += width)
    {
        typename P::packet_t px;
        typename P::packet_t py;
        P::loadu(x + c, px);
        P::loadu(y + c, py);
        P::fma(pa, px, py, py);
        P::storeu(py, y + c);
    }
    for (; c < len; ++c)
    {
        y[c] += alpha * x[c];
    }
#else
    for (linalg_long c = 0; c < len; ++c)
    {
        y[c] += alpha * x[c];
    }
#endif
}

template <typename T> size_t qr_size(linalg_long rows, linalg_long columns, size_t scratch)
{
    LINALG_CHECK(rows > 0 && columns > 0, "QR: dimensions must be positive");
    detail::workspace_arena arena;
    arena.next<T>(scratch);
    return arena.total_size();
}

template <typename T>
void qr_r(linalg_long rows,
    linalg_long       columns,
    T*                A,
    linalg_long       lda,
    T*                tau,
    void*             workspace,
    size_t            workspace_size)
{
    LINALG_CHECK(A && tau && workspace, "QR apply_r: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_r: invalid row stride");
    LINALG_CHECK(
        workspace_size >= qr_size<T>(rows, columns, columns), "QR apply_r: workspace too small");
    const auto              k = std::min(rows, columns);
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   w = arena.next<T>(static_cast<size_t>(columns));
    // LAPACK storage: diagonal/upper triangle are R; each reflector has an
    // implicit leading 1 and its remaining entries below A's diagonal.
    for (linalg_long i = 0; i < k; ++i)
    {
        const T alpha     = A[i * lda + i];
        T       tail_norm = T(0);
        for (linalg_long r = i + 1; r < rows; ++r)
        {
            tail_norm = std::hypot(tail_norm, A[r * lda + i]);
        }
        if (tail_norm == T(0))
        {
            tau[i] = T(0);
            continue;
        }
        const T beta = -std::copysign(std::hypot(alpha, tail_norm), alpha);
        tau[i]       = (beta - alpha) / beta;
        for (linalg_long r = i + 1; r < rows; ++r){
            A[r * lda + i] /= alpha - beta;
        }
        A[i * lda + i]      = beta;
        const auto trailing = columns - i - 1;
        std::copy_n(A + i * lda + i + 1, static_cast<size_t>(trailing), w);
        for (linalg_long r = i + 1; r < rows; ++r){
            qr_axpy(w, A + r * lda + i + 1, A[r * lda + i], trailing);
        }
        qr_axpy(A + i * lda + i + 1, w, -tau[i], trailing);
        for (linalg_long r = i + 1; r < rows; ++r){
            qr_axpy(A + r * lda + i + 1, w, -tau[i] * A[r * lda + i], trailing);
        }
    }
}

template <typename T>
void qr_q(linalg_long rows,
    linalg_long       columns,
    T*                A,
    linalg_long       lda,
    const T*          tau,
    void*             workspace,
    size_t            workspace_size)
{
    const auto k = std::min(rows, columns);
    LINALG_CHECK(A && tau && workspace, "QR apply_q: A, tau, workspace required");
    LINALG_CHECK(lda >= columns, "QR apply_q: invalid row stride");
    LINALG_CHECK(workspace_size >= qr_size<T>(rows, columns, k), "QR apply_q: workspace too small");
    detail::workspace_arena arena(workspace, workspace_size);
    auto*                   w = arena.next<T>(static_cast<size_t>(k));
    // Form the first k columns of Q in place, from the last reflector to the first.
    for (linalg_long step = 0; step < k; ++step)
    {
        const auto i        = k - 1 - step;
        const auto trailing = k - i - 1;
        std::copy_n(A + i * lda + i + 1, static_cast<size_t>(trailing), w);
        for (linalg_long r = i + 1; r < rows; ++r){
            qr_axpy(w, A + r * lda + i + 1, A[r * lda + i], trailing);
        }
        qr_axpy(A + i * lda + i + 1, w, -tau[i], trailing);
        for (linalg_long r = i + 1; r < rows; ++r)
        {
            qr_axpy(A + r * lda + i + 1, w, -tau[i] * A[r * lda + i], trailing);
            A[r * lda + i] *= -tau[i];
        }
        A[i * lda + i] = T(1) - tau[i];
        for (linalg_long r = 0; r < i; ++r){
            A[r * lda + i] = T(0);}
    }
}
}  // namespace
template <>
size_t qr_decomposition<float, linalg_backend::NATIVE>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    return qr_size<float>(rows, columns, columns);
}
template <>
void qr_decomposition<float, linalg_backend::NATIVE>::apply_r(linalg_long rows,
    linalg_long                                                           columns,
    float*                                                                A,
    linalg_long                                                           lda,
    float*                                                                tau,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<float>/apply_r");
    qr_r(rows, columns, A, lda, tau, workspace, workspace_size);
}
template <>
size_t qr_decomposition<float, linalg_backend::NATIVE>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    return qr_size<float>(rows, columns, std::min(rows, columns));
}
template <>
void qr_decomposition<float, linalg_backend::NATIVE>::apply_q(linalg_long rows,
    linalg_long                                                           columns,
    float*                                                                A,
    linalg_long                                                           lda,
    const float*                                                          tau,
    void*                                                                 workspace,
    size_t                                                                workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<float>/apply_q");
    qr_q(rows, columns, A, lda, tau, workspace, workspace_size);
}
template <>
size_t qr_decomposition<double, linalg_backend::NATIVE>::workspace_size_r(
    linalg_long rows, linalg_long columns)
{
    return qr_size<double>(rows, columns, columns);
}
template <>
void qr_decomposition<double, linalg_backend::NATIVE>::apply_r(linalg_long rows,
    linalg_long                                                            columns,
    double*                                                                A,
    linalg_long                                                            lda,
    double*                                                                tau,
    void*                                                                  workspace,
    size_t                                                                 workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<double>/apply_r");
    qr_r(rows, columns, A, lda, tau, workspace, workspace_size);
}
template <>
size_t qr_decomposition<double, linalg_backend::NATIVE>::workspace_size_q(
    linalg_long rows, linalg_long columns)
{
    return qr_size<double>(rows, columns, std::min(rows, columns));
}
template <>
void qr_decomposition<double, linalg_backend::NATIVE>::apply_q(linalg_long rows,
    linalg_long                                                            columns,
    double*                                                                A,
    linalg_long                                                            lda,
    const double*                                                          tau,
    void*                                                                  workspace,
    size_t                                                                 workspace_size,
    void* /*stream*/)
{
    LINALG_SCOPE("qr_decomposition/native<double>/apply_q");
    qr_q(rows, columns, A, lda, tau, workspace, workspace_size);
}
}  // namespace linalg
