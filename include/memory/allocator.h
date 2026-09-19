/*
 * LinearAlgebra — minimal scratch-buffer allocator.
 *
 * Reconstructed stand-in for the (orphaned) memory/allocator.h this code
 * used to depend on. matrix_operation only ever needs a pair of static
 * allocate(n)/free(p) entry points for untyped scratch buffers (packed GEMM
 * panels, SVD/inversion work arrays) — it never rebinds, copy-constructs, or
 * otherwise treats this as a full C++ Allocator-concept type.
 *
 * Design note (PyTorch / Eigen3 as reference points, per project convention):
 * PyTorch's c10::Allocator is a virtual interface returning a DataPtr with a
 * custom deleter, so several backends (CPU/CUDA/...) can be swapped behind
 * one pointer type; Eigen3's internal aligned_allocator (Eigen/src/Core/
 * util/Memory.h) instead exposes plain static allocate_aligned/
 * deallocate_aligned free functions with no virtual dispatch, since Eigen
 * only ever targets host memory. This header follows Eigen's simpler,
 * static-dispatch shape — matrix_operation is host-only scalar/BLAS code
 * with no device abstraction to plug in, so PyTorch's runtime-polymorphic
 * DataPtr machinery would be pure overhead here.
 */
#pragma once

#include <cstddef>
#include <new>

namespace linalg
{

template <typename T>
struct allocator
{
    static T* allocate(std::size_t n)
    {
        if (n == 0)
        {
            return nullptr;
        }
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    static void free(T* p) noexcept { ::operator delete(p); }
};

}  // namespace linalg
