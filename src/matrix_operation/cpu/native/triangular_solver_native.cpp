#include "include/matrix_operation/triangular_solver.h"

#include <algorithm>

#include "include/common/linalg_profiler.h"
#include "include/common/workspace_arena.h"
#include "include/matrix_operation/multiplication.h"
#include "src/common/triangular_solver_checks.h"
#if LINALG_HAS_VECTORIZATION
#include <backend/simd.h>
#endif

namespace linalg
{
namespace
{
constexpr linalg_int triangular_block = 32;

// Independent RHS columns form SIMD lanes; substitution order is unchanged.
template <typename T> void subtract_row(T* y, const T* x, T scale, linalg_int count)
{
    linalg_int j = 0;
#if LINALG_HAS_VECTORIZATION
    using P = simd<T>;
    typename P::simd_t alpha;
    alpha = P::set(-scale);
    for (; j + P::size <= count; j += P::size)
    {
        typename P::simd_t px, py;
        px = P::loadu(x + j);
        py = P::loadu(y + j);
        py = P::fma(alpha, px, py);
        P::storeu(py, y + j);
    }
#endif
    for (; j < count; ++j)
    {
        y[j] -= scale * x[j];
    }
}

template <typename T> size_t gemm_scratch(linalg_int n, linalg_int nrhs)
{
    size_t bytes = 0;
    for (linalg_int remaining = n; remaining > triangular_block; remaining -= triangular_block)
    {
        bytes = std::max(bytes,
            multiplication<T, linalg_backend::NATIVE>::workspace_size(
                remaining - triangular_block, nrhs, triangular_block));
    }
    return bytes;
}

template <typename T> size_t size_impl(linalg_int n, linalg_int nrhs)
{
    detail::check_triangular_dimensions(n, nrhs);
    if (n <= triangular_block)
    {
        return 0;
    }
    detail::workspace_arena arena;
    arena.next<T>(static_cast<size_t>(n) * static_cast<size_t>(nrhs));
    arena.next<unsigned char>(gemm_scratch<T>(n, nrhs));
    return arena.total_size();
}

template <typename T>
void solve(bool upper,
    bool        transpose,
    bool        unit,
    linalg_int  n,
    linalg_int  nrhs,
    const T*    a,
    linalg_int  lda,
    const T*    b,
    linalg_int  ldb,
    T*          x,
    linalg_int  ldx,
    void*       workspace,
    size_t      workspace_size)
{
    detail::check_triangular_arguments(a, b, x, n, nrhs, lda, ldb, ldx);
    const auto required = size_impl<T>(n, nrhs);
    LINALG_CHECK(workspace_size >= required && (required == 0 || workspace),
        "triangular_solver: workspace is null or too small");
    detail::workspace_arena arena(workspace, workspace_size);
    T* update = n > triangular_block ? arena.next<T>(static_cast<size_t>(n) * nrhs) : nullptr;
    const auto gemm_bytes = n > triangular_block ? gemm_scratch<T>(n, nrhs) : 0;
    auto*      gemm       = n > triangular_block ? arena.next<unsigned char>(gemm_bytes) : nullptr;
    for (linalg_int i = 0; i < n; ++i)
    {
        std::copy_n(b + i * ldb, nrhs, x + i * ldx);
    }
    const auto coefficient = [&](linalg_int i, linalg_int j)
    { return transpose ? a[j * lda + i] : a[i * lda + j]; };
    const bool effective_upper = upper != transpose;
    for (linalg_int done = 0; done < n; done += triangular_block)
    {
        const linalg_int width = std::min(triangular_block, n - done);
        const linalg_int begin = effective_upper ? n - done - width : done;
        const linalg_int end   = begin + width;
        for (linalg_int step = 0; step < width; ++step)
        {
            const linalg_int i   = effective_upper ? end - step - 1 : begin + step;
            T*               row = x + i * ldx;
            for (linalg_int k = effective_upper ? i + 1 : begin; k < (effective_upper ? end : i);
                ++k)
            {
                subtract_row(row, x + k * ldx, coefficient(i, k), nrhs);
            }
            if (!unit)
            {
                const T    diagonal = coefficient(i, i);
                linalg_int j        = 0;
#if LINALG_HAS_VECTORIZATION
                using P = simd<T>;
                typename P::simd_t d;
                d = P::set(diagonal);
                for (; j + P::size <= nrhs; j += P::size)
                {
                    typename P::simd_t value;
                    value = P::loadu(row + j);
                    value = P::div(value, d);
                    P::storeu(value, row + j);
                }
#endif
                for (; j < nrhs; ++j)
                {
                    row[j] /= diagonal;
                }
            }
        }
        const linalg_int first = effective_upper ? 0 : end;
        const linalg_int rows  = effective_upper ? begin : n - end;
        if (rows == 0)
        {
            continue;
        }
        const T* block = transpose ? a + begin * lda + first : a + first * lda + begin;
        multiplication<T, linalg_backend::NATIVE>::apply(transpose,
            false,
            rows,
            nrhs,
            width,
            block,
            lda,
            x + begin * ldx,
            ldx,
            update,
            nrhs,
            gemm,
            gemm_bytes);
        for (linalg_int r = 0; r < rows; ++r)
        {
            subtract_row(x + (first + r) * ldx, update + r * nrhs, T(1), nrhs);
        }
    }
}
}  // namespace

template <typename T, linalg_backend Backend>
void triangular_solver<T, Backend>::apply(bool upper,
    bool                                       transpose,
    bool                                       unit,
    linalg_int                                 n,
    linalg_int                                 nrhs,
    const T*                                   a,
    linalg_int                                 lda,
    const T*                                   b,
    linalg_int                                 ldb,
    T*                                         x,
    linalg_int                                 ldx,
    void*                                      workspace,
    size_t                                     workspace_size,
    void* /*stream*/)
{
    LINALG_FUNCTION();
    solve(upper, transpose, unit, n, nrhs, a, lda, b, ldb, x, ldx, workspace, workspace_size);
}

template <typename T, linalg_backend Backend>
size_t triangular_solver<T, Backend>::workspace_size(linalg_int n, linalg_int nrhs)
{
    return size_impl<T>(n, nrhs);
}

template struct triangular_solver<float, linalg_backend::NATIVE>;
template struct triangular_solver<double, linalg_backend::NATIVE>;
}  // namespace linalg
