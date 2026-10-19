/*
 * LinearAlgebra — test-only dense matrix helper.
 *
 * Replaces the (orphaned) terminals/matrix.h type the original matrix_operation
 * tests were written against. matrix_operation's public API is plain
 * row-major raw-pointer C (T* data, lda) — it never depended on a matrix
 * class — so this type exists purely to give the ported GoogleTest fixtures
 * somewhere convenient to build/own test data and hand `.data()` to the
 * library calls; it is not part of the LinearAlgebra library itself.
 *
 * Design note (PyTorch / Eigen3 as reference points, per project convention):
 * both PyTorch (at::Tensor) and Eigen3 (Eigen::Matrix) separate storage
 * layout/strides from the numerics that operate on them, and neither is a
 * fit here — this header intentionally stays a bare row-major
 * std::vector<double>-backed accessor, no expression templates, no strides,
 * no generality beyond what the ported fixtures need, since matrix_operation
 * itself already owns all of the actual linear-algebra logic.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/transpose.h"

namespace linalg_test
{

template <typename T>
class dense_matrix
{
public:
    dense_matrix() : rows_(0), cols_(0) {}

    dense_matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, T{})
    {
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    T& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    T*       data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    T*       begin() { return data_.data(); }
    T*       end() { return data_.data() + data_.size(); }
    const T* begin() const { return data_.data(); }
    const T* end() const { return data_.data() + data_.size(); }

    void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

private:
    std::size_t    rows_;
    std::size_t    cols_;
    std::vector<T> data_;
};

template <typename T>
dense_matrix<T> transpose(const dense_matrix<T>& a)
{
    // Apply in-place transpose to a copy of a's flat data, then move the
    // result into a new dense_matrix with swapped dimensions so that the
    // row-major accessor (i*cols_+j) uses the correct stride.
    dense_matrix<T> scratch(a);  // copy — same rows/cols as a
    const auto      ws_size = linalg::transpose<T, linalg::linalg_backend::native>::workspace_size(
        static_cast<linalg_long>(a.rows()), static_cast<linalg_long>(a.cols()));
    std::vector<unsigned char> ws(ws_size);
    linalg::transpose<T, linalg::linalg_backend::native>::apply(
        static_cast<linalg_long>(a.rows()),
        static_cast<linalg_long>(a.cols()),
        scratch.data(),
        ws.data(),
        ws.size());
    // scratch.data() now holds the transposed layout (cols x rows); wrap it
    // in a dense_matrix with the correct (swapped) dimension metadata.
    dense_matrix<T> t(a.cols(), a.rows());
    std::copy(scratch.data(), scratch.data() + a.rows() * a.cols(), t.data());
    return t;
}

template <typename T>
dense_matrix<T> matmul(const dense_matrix<T>& a, const dense_matrix<T>& b)
{
    dense_matrix<T> c(a.rows(), b.cols());
    const auto      ws_size =
        linalg::multiplication<T, linalg::linalg_backend::native>::workspace_size(
            static_cast<linalg_int>(a.rows()),
            static_cast<linalg_int>(b.cols()),
            static_cast<linalg_int>(a.cols()));
    std::vector<unsigned char> ws(ws_size);
    linalg::multiplication<T, linalg::linalg_backend::native>::apply(false,
        false,
        static_cast<linalg_int>(a.rows()),
        static_cast<linalg_int>(b.cols()),
        static_cast<linalg_int>(a.cols()),
        a.data(),
        static_cast<linalg_int>(a.cols()),
        b.data(),
        static_cast<linalg_int>(b.cols()),
        c.data(),
        static_cast<linalg_int>(c.cols()),
        ws.data(),
        ws.size());
    return c;
}

}  // namespace linalg_test
