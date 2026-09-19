#include "include/matrix_operation/matrix_transpose.h"

//#include <stdexcept>
#include "include/common/configure.h"  // IWYU pragma: keep
#include "include/common/macros.h"
#include "include/util/exception.h"

#if defined(LINALG_VECTORIZED)
#include <cmath>
#include <limits>

#include "common/packet.h"
#endif  // LINALG_VECTORIZED

#ifdef LINALG_ENABLE_MKL
#include <mkl.h>
#else
#include <algorithm>  // std::swap (until C++11)
#include <iostream>   // std::cout
#include <iterator>   // std::ostream_iterator
#include <vector>
#endif

namespace linalg
{
namespace
{
#ifndef LINALG_ENABLE_MKL
//fixme!
//#if defined(LINALG_VECTORIZED)
//template <typename value_t, quarisma_int block_size>
//void transpose_block_vectorized(
//    value_t* A, quarisma_int rows, quarisma_int columns, quarisma_int i, quarisma_int j)
//{
//    using simd_t = typename simd<value_t>::simd_t;
//    simd_t tmp[block_size];
//
//    for (quarisma_int k = 0; k < block_size; ++k)
//    {
//        simd<value_t>::loadu(&A[(i + k) * columns + j], tmp[k]);
//    }
//
//    simd<value_t>::template ptranspose<block_size>(tmp);
//
//    for (quarisma_int k = 0; k < block_size; ++k)
//    {
//        simd<value_t>::storeu(tmp[k], &A[(j + k) * rows + i]);
//    }
//}
//#endif
//
//template <typename value_t, quarisma_int block_size>
//void transpose_block_vectorized(value_t* A, quarisma_int rows, quarisma_int columns)
//{
//    std::vector<value_t> temp(block_size * block_size);
//
//    for (quarisma_int i = 0; i < rows; i += block_size)
//    {
//        for (quarisma_int j = 0; j < columns; j += block_size)
//        {
//            quarisma_int ib = std::min(block_size, rows - i);
//            quarisma_int jb = std::min(block_size, columns - j);
//
//#if defined(LINALG_VECTORIZED)
//            if (ib == block_size && jb == block_size)
//            {
//                transpose_block_vectorized<value_t, block_size>(A, rows, columns, i, j);
//            }
//            else
//#endif
//            {
//                // Copy block to temp
//                for (quarisma_int k = 0; k < ib; ++k)
//                {
//                    for (quarisma_int l = 0; l < jb; ++l)
//                    {
//                        temp[k * block_size + l] = A[(i + k) * columns + (j + l)];
//                    }
//                }
//
//                // Copy transposed block back
//                for (quarisma_int k = 0; k < jb; ++k)
//                {
//                    for (quarisma_int l = 0; l < ib; ++l)
//                    {
//                        A[(j + k) * rows + (i + l)] = temp[l * block_size + k];
//                    }
//                }
//            }
//        }
//    }
//}

template <class RandomIterator>
void transpose(RandomIterator first, RandomIterator last, int m)
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
#endif  // !LINALG_ENABLE_MKL

template <typename T>
void mkl_transpose(
    LINALG_UNUSED quarisma_long rows, LINALG_UNUSED quarisma_long columns, LINALG_UNUSED T* a)
{
    throw std::runtime_error("MKL transpose is not implemented for this type.");
}

#ifdef LINALG_ENABLE_MKL
template <>
void mkl_transpose<double>(quarisma_long rows, quarisma_long columns, double* a)
{
    mkl_dimatcopy('R', 'T', rows, columns, 1., a, columns, rows);
}

template <>
void mkl_transpose<float>(quarisma_long rows, quarisma_long columns, float* a)
{
    mkl_simatcopy('R', 'T', rows, columns, 1., a, columns, rows);
}
#endif  // defined

template <typename T>
void matrix_transpose(quarisma_long rows, quarisma_long columns, T* m)
{
#ifdef LINALG_ENABLE_MKL
    mkl_transpose<T>(rows, columns, m);

//#elif defined(LINALG_VECTORIZED)
//    transpose_block_vectorized<T, simd<T>::size>(
//        m, static_cast<quarisma_int>(rows), static_cast<quarisma_int>(columns));
#else
    transpose<T*>(m, m + rows * columns, static_cast<quarisma_int>(columns));
#endif  // __MKL_
}
}  // namespace

void matrix_transpose(quarisma_long rows, quarisma_long columns, float* m)
{
    matrix_transpose<float>(rows, columns, m);
}

void matrix_transpose(quarisma_long rows, quarisma_long columns, double* m)
{
    matrix_transpose<double>(rows, columns, m);
}
}  // namespace linalg
