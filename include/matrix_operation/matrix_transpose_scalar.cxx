#include <algorithm>
#include <vector>

#include "include/matrix_operation/matrix_transpose_dispatch.h"

namespace linalg
{
namespace detail
{
namespace
{

template <class RandomIterator>
void transpose_cycles(RandomIterator first, RandomIterator last, int m)
{
    const int         mn1 = (last - first - 1);
    const int         n   = (last - first) / m;
    std::vector<bool> visited(last - first);
    RandomIterator    cycle = first;
    while (++cycle != last)
    {
        if (visited[cycle - first])
        {
            continue;
        }
        int a = cycle - first;
        do
        {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

template <typename T>
void transpose_scalar_impl(quarisma_long rows, quarisma_long columns, T* m)
{
    transpose_cycles<T*>(m, m + rows * columns, static_cast<quarisma_int>(columns));
}

}  // namespace

void transpose_scalar_f32(quarisma_long rows, quarisma_long columns, float* m)
{
    transpose_scalar_impl(rows, columns, m);
}

void transpose_scalar_f64(quarisma_long rows, quarisma_long columns, double* m)
{
    transpose_scalar_impl(rows, columns, m);
}

LINALG_REGISTER_DISPATCH(transpose_f32_stub, scalar, transpose_scalar_f32);
LINALG_REGISTER_DISPATCH(transpose_f64_stub, scalar, transpose_scalar_f64);

}  // namespace detail
}  // namespace linalg
