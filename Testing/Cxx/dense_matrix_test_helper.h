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

}  // namespace linalg_test
