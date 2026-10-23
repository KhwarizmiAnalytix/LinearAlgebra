#pragma once

#include <cstdint>
#include <limits>

#include "include/common/linalg_checks.h"
#include "include/common/narrow.h"
#include "include/matrix_operation/schur_decomposition.h"

namespace linalg::detail
{
inline int schur_decomposition_order(linalg_int n)
{
    LINALG_CHECK(n > 0, "schur_decomposition: n must be positive");
    const int order = checked_narrow<int>(n);
    LINALG_CHECK(order <= std::numeric_limits<int>::max() / order,
        "schur_decomposition: matrix exceeds supported index range");
    return order;
}

template <typename T> size_t portable_schur_workspace_size(linalg_int n)
{
    const auto order = static_cast<size_t>(schur_decomposition_order(n));
    return (4 * order * order + 2 * order) * sizeof(T);
}

template <typename T, linalg_backend Backend>
void check_schur_decomposition(const T* a,
    linalg_int                          n,
    linalg_int                          lda,
    T*                                  sr,
    T*                                  si,
    linalg_int                          lds,
    T*                                  qr,
    T*                                  qi,
    linalg_int                          ldq,
    void*                               workspace,
    size_t                              bytes)
{
    const int   order      = schur_decomposition_order(n);
    const void* pointers[] = {a, sr, si, qr, qi, workspace};
    for (int i = 0; i < 6; ++i)
    {
        LINALG_CHECK(pointers[i] != nullptr, "schur_decomposition: null buffer");
        for (int j = i + 1; j < 6; ++j)
        {
            LINALG_CHECK(
                pointers[i] != pointers[j], "schur_decomposition: buffers must be distinct");
        }
    }
    LINALG_CHECK(lda >= n && lds >= n && ldq >= n, "schur_decomposition: invalid row stride");
    const auto limit = std::numeric_limits<int>::max() / order;
    LINALG_CHECK(lda <= limit && lds <= limit && ldq <= limit,
        "schur_decomposition: row stride span exceeds supported indexing");
    LINALG_CHECK(bytes >= (schur_decomposition<T, Backend>::workspace_size(n)),
        "schur_decomposition: workspace too small");
    LINALG_CHECK(reinterpret_cast<std::uintptr_t>(workspace) % alignof(T) == 0,
        "schur_decomposition: workspace must be aligned for its scalar type");
}
}  // namespace linalg::detail
