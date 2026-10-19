#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "test_backend.h"
#include "test_reporting.h"
#include "gtest/gtest.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct QRTolerance;
template <> struct QRTolerance<float>
{
    static constexpr double tolerance = 1e-3;
};
template <> struct QRTolerance<double>
{
    static constexpr double tolerance = 1e-12;
};

template <typename T, linalg::linalg_backend Backend>
void test_qr(std::size_t rows, std::size_t columns, std::size_t padding = 0, bool zero = false)
{
    using QR                 = linalg::qr_decomposition<T, Backend>;
    const auto      k        = std::min(rows, columns);
    constexpr T     sentinel = T(12345);
    dense_matrix<T> A(rows, columns + padding);
    A.fill(sentinel);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            A(i, j) = zero ? T(0) : static_cast<T>(20. * rand() / RAND_MAX - 10.);
    const auto                 original = A;
    std::vector<T>             tau(k);
    std::vector<unsigned char> r_scratch(QR::workspace_size_r(rows, columns));
    QR::apply_r(rows, columns, A.begin(), A.cols(), tau.data(), r_scratch.data(), r_scratch.size());

    // The client chooses to save R before Q formation overwrites the packed matrix.
    dense_matrix<T> R(k, columns);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < columns; ++j)
            R(i, j) = A(i, j);
    const auto packed    = A;
    const auto saved_tau = tau;

    // R-only needs only the factorization phase and its own scratch size.
    auto           r_only = original;
    std::vector<T> tau_only(k);
    QR::apply_r(rows,
        columns,
        r_only.begin(),
        r_only.cols(),
        tau_only.data(),
        r_scratch.data(),
        r_scratch.size());
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < columns; ++j)
            EXPECT_EQ(r_only(i, j), R(i, j));

    // Only packed A and tau carry state; factorization scratch may be discarded.
    std::fill(r_scratch.begin(), r_scratch.end(), 0xff);
    std::vector<unsigned char> q_scratch(QR::workspace_size_q(rows, columns));
    QR::apply_q(rows, columns, A.begin(), A.cols(), tau.data(), q_scratch.data(), q_scratch.size());
    EXPECT_EQ(tau, saved_tau);

    T max_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
        {
            T sum = 0;
            for (std::size_t c = 0; c < k; ++c)
                sum += A(i, c) * R(c, j);
            max_error = std::max(max_error, static_cast<T>(std::fabs(original(i, j) - sum)));
        }
    EXPECT_LT(max_error, static_cast<T>(QRTolerance<T>::tolerance));

    T max_orth_error = 0;
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
        {
            T sum = 0;
            for (std::size_t r = 0; r < rows; ++r)
                sum += A(r, i) * A(r, j);
            const T expected = i == j ? T(1) : T(0);
            max_orth_error   = std::max(max_orth_error, static_cast<T>(std::fabs(sum - expected)));
        }
    EXPECT_LT(max_orth_error, static_cast<T>(QRTolerance<T>::tolerance));

    // Q formation touches only the first k columns; padded strides stay intact.
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = k; j < A.cols(); ++j)
            EXPECT_EQ(A(i, j), packed(i, j));
}
}  // namespace

TEST(Math, QRDecomposition)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            for (const std::size_t padding : {0, 3})
            {
                test_qr<float, Backend>(6, 5, padding);
                test_qr<float, Backend>(5, 6, padding);
                test_qr<float, Backend>(4, 4, padding);
                test_qr<double, Backend>(6, 5, padding);
                test_qr<double, Backend>(5, 6, padding);
                test_qr<double, Backend>(4, 4, padding);
                test_qr<float, Backend>(1, 5, padding);
                test_qr<double, Backend>(5, 1, padding);
                test_qr<double, Backend>(1, 1, padding);
                test_qr<float, Backend>(6, 5, padding, true);
                test_qr<double, Backend>(5, 6, padding, true);
            }
        });
}

TEST(Math, QRDecompositionInvalidInputs)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            using QR               = linalg::qr_decomposition<double, Backend>;
            std::vector<double>        A(12), tau(3);
            std::vector<unsigned char> ws(QR::workspace_size(4, 3));
            EXPECT_ANY_THROW(QR::apply_r(4, 3, nullptr, 3, tau.data(), ws.data(), ws.size()));
            EXPECT_ANY_THROW(QR::apply_r(4, 3, A.data(), 2, tau.data(), ws.data(), ws.size()));
            EXPECT_ANY_THROW(QR::apply_r(4, 3, A.data(), 3, nullptr, ws.data(), ws.size()));
            EXPECT_ANY_THROW(QR::apply_r(4, 3, A.data(), 3, tau.data(), ws.data(), 0));
            EXPECT_ANY_THROW(QR::workspace_size_r(0, 3));
            EXPECT_ANY_THROW(QR::workspace_size_q(4, 0));
            QR::apply_r(4, 3, A.data(), 3, tau.data(), ws.data(), ws.size());
            EXPECT_ANY_THROW(QR::apply_q(4, 3, A.data(), 3, nullptr, ws.data(), ws.size()));
            EXPECT_ANY_THROW(QR::apply_q(4, 3, A.data(), 2, tau.data(), ws.data(), ws.size()));
            EXPECT_ANY_THROW(QR::apply_q(4, 3, A.data(), 3, tau.data(), ws.data(), 0));
        });
}
