#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/eigen.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "test_backend.h"
#include "gtest/gtest.h"

namespace
{
using linalg_test::dense_matrix;

template <typename> struct EigenTolerance
{
};

template <> struct EigenTolerance<float>
{
    static constexpr double tolerance = 2.e-3;
};

template <> struct EigenTolerance<double>
{
    static constexpr double tolerance = 1.e-8;
};

template <typename value_t, linalg::linalg_backend Backend>
std::vector<unsigned char> symmetric_eigen_workspace(linalg_int n)
{
    const auto bytes = linalg::symmetric_eigen_backend<value_t, Backend>::workspace_size(n);
    return std::vector<unsigned char>(bytes);
}

template <typename value_t, linalg::linalg_backend Backend>
std::vector<unsigned char> general_eigen_workspace(linalg_int n, bool compute_eigenvectors)
{
    const auto bytes =
        linalg::general_eigen_backend<value_t, Backend>::workspace_size(n, compute_eigenvectors);
    return std::vector<unsigned char>(bytes);
}

// Symmetric case: A = Q * diag(known eigenvalues) * Q^T for a random
// orthogonal Q (built from qr_decomposition). Eigenvalues must come back
// sorted ascending and V^T * A * V must reconstruct the diagonal.
template <typename value_t, linalg::linalg_backend Backend> void test_symmetric_eigen(std::size_t n)
{
    dense_matrix<value_t> R0(n, n);
    for (std::size_t i = 0; i < n * n; i++)
    {
        R0.begin()[i] = static_cast<value_t>(2. * rand() / RAND_MAX - 1.);
    }
    dense_matrix<value_t> Q(n, n);
    dense_matrix<value_t> R(n, n);
    const auto            qr_ws_size = linalg::qr_decomposition<value_t, Backend>::workspace_size(
        static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    std::vector<unsigned char> qr_ws(qr_ws_size);
    linalg::qr_decomposition<value_t, Backend>::apply(static_cast<linalg_long>(n),
        static_cast<linalg_long>(n),
        R0.begin(),
        static_cast<linalg_long>(n),
        Q.begin(),
        static_cast<linalg_long>(n),
        R.begin(),
        static_cast<linalg_long>(n),
        qr_ws.data(),
        qr_ws_size);

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; i++)
    {
        known[i] = static_cast<value_t>(i + 1);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; k++)
            {
                sum += Q(i, k) * known[k] * Q(j, k);
            }
            A(i, j) = sum;
        }
    }

    std::vector<value_t>  evals(n);
    dense_matrix<value_t> evecs(n, n);
    auto       eigen_ws = symmetric_eigen_workspace<value_t, Backend>(static_cast<linalg_int>(n));
    const bool ok       = linalg::symmetric_eigen_backend<value_t, Backend>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        evals.data(),
        evecs.begin(),
        static_cast<linalg_int>(n),
        eigen_ws.data(),
        eigen_ws.size());
    EXPECT_TRUE(ok);

    for (std::size_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(static_cast<double>(evals[i]),
            static_cast<double>(known[i]),
            EigenTolerance<value_t>::tolerance);
    }

    // AV = A * evecs, then check evecs^T * AV == diag(evals).
    dense_matrix<value_t> AV(n, n);
    for (std::size_t p = 0; p < n; p++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t q = 0; q < n; q++)
            {
                sum += A(p, q) * evecs(q, j);
            }
            AV(p, j) = sum;
        }
    }

    value_t max_error = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t p = 0; p < n; p++)
            {
                sum += evecs(p, i) * AV(p, j);
            }
            const value_t expected = (i == j) ? evals[i] : value_t(0);
            max_error = std::max(max_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
    }
    EXPECT_LT(max_error, static_cast<value_t>(EigenTolerance<value_t>::tolerance));
}

// General (non-symmetric) case with a known real spectrum: A = P * D * P^-1
// for a diagonally-dominant (hence invertible) random P.
template <typename value_t, linalg::linalg_backend Backend>
void test_general_eigen_real_spectrum(std::size_t n)
{
    dense_matrix<value_t> P(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            P(i, j) = static_cast<value_t>(2. * rand() / RAND_MAX - 1.);
        }
        P(i, i) += static_cast<value_t>(5);
    }

    dense_matrix<value_t> Pinv(n, n);
    for (std::size_t i = 0; i < n * n; i++)
    {
        Pinv.begin()[i] = P.begin()[i];
    }
    std::vector<linalg_int> pivot(n);
    const auto              invert_ws_size = linalg::inversion<value_t, Backend>::workspace_size(
        static_cast<linalg_int>(n), linalg::linear_solver_type::LU_LINEAR_SOLVER);
    std::vector<unsigned char> invert_ws(invert_ws_size);
    linalg::inversion<value_t, Backend>::apply(Pinv.begin(),
        pivot.data(),
        static_cast<linalg_int>(n),
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        invert_ws.data(),
        invert_ws_size);

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; i++)
    {
        known[i] = static_cast<value_t>(2 * static_cast<int>(i) - 3);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; k++)
            {
                sum += P(i, k) * known[k] * Pinv(k, j);
            }
            A(i, j) = sum;
        }
    }

    std::vector<value_t> er(n);
    std::vector<value_t> ei(n);
    auto eigen_ws = general_eigen_workspace<value_t, Backend>(static_cast<linalg_int>(n), false);
    const bool ok = linalg::general_eigen_backend<value_t, Backend>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        er.data(),
        ei.data(),
        nullptr,
        static_cast<linalg_int>(n),
        eigen_ws.data(),
        eigen_ws.size());
    EXPECT_TRUE(ok);

    std::vector<value_t> got = er;
    std::sort(got.begin(), got.end());
    for (std::size_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(static_cast<double>(got[i]),
            static_cast<double>(known[i]),
            EigenTolerance<value_t>::tolerance);
        EXPECT_NEAR(static_cast<double>(ei[i]), 0.0, EigenTolerance<value_t>::tolerance);
    }
}

// Complex-conjugate pair: a 2x2 rotation-scale block [[a,-b],[b,a]] has
// eigenvalues a +- b*i exactly.
template <typename value_t, linalg::linalg_backend Backend> void test_general_eigen_complex_pair()
{
    const value_t         a = static_cast<value_t>(1.5);
    const value_t         b = static_cast<value_t>(2.0);
    dense_matrix<value_t> A(2, 2);
    A(0, 0) = a;
    A(0, 1) = -b;
    A(1, 0) = b;
    A(1, 1) = a;

    value_t    er[2];
    value_t    ei[2];
    auto       eigen_ws = general_eigen_workspace<value_t, Backend>(2, false);
    const bool ok       = linalg::general_eigen_backend<value_t, Backend>::apply(
        A.begin(), 2, 2, er, ei, nullptr, 2, eigen_ws.data(), eigen_ws.size());
    EXPECT_TRUE(ok);

    EXPECT_NEAR(
        static_cast<double>(er[0]), static_cast<double>(a), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(
        static_cast<double>(er[1]), static_cast<double>(a), EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(std::fabs(ei[0])),
        static_cast<double>(b),
        EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(std::fabs(ei[1])),
        static_cast<double>(b),
        EigenTolerance<value_t>::tolerance);
    EXPECT_NEAR(static_cast<double>(ei[0] + ei[1]), 0.0, EigenTolerance<value_t>::tolerance);
}

// Real-eigenvector residual check on a lower-triangular matrix (eigenvalues
// are exactly its diagonal entries).
template <typename value_t, linalg::linalg_backend Backend> void test_general_eigen_eigenvectors()
{
    dense_matrix<value_t> A(3, 3);
    A(0, 0) = 2;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(2, 1) = 1;
    A(2, 2) = 4;

    value_t               er[3];
    value_t               ei[3];
    dense_matrix<value_t> evecs(3, 3);
    auto                  eigen_ws = general_eigen_workspace<value_t, Backend>(3, true);
    const bool            ok       = linalg::general_eigen_backend<value_t, Backend>::apply(
        A.begin(), 3, 3, er, ei, evecs.begin(), 3, eigen_ws.data(), eigen_ws.size());
    EXPECT_TRUE(ok);

    for (std::size_t i = 0; i < 3; i++)
    {
        value_t max_error = 0;
        for (std::size_t r = 0; r < 3; r++)
        {
            value_t sum = 0;
            for (std::size_t c = 0; c < 3; c++)
            {
                sum += A(r, c) * evecs(c, i);
            }
            const value_t expected = er[i] * evecs(r, i);
            max_error = std::max(max_error, static_cast<value_t>(std::fabs(sum - expected)));
        }
        EXPECT_LT(max_error, static_cast<value_t>(EigenTolerance<value_t>::tolerance));
    }
}

// Results from one backend, kept so later backends can be compared to the
// first (native) one.
template <typename value_t> struct eigen_result
{
    std::vector<value_t>  evals;
    std::vector<value_t>  ei;
    dense_matrix<value_t> evecs;
    explicit eigen_result(std::size_t n) : evals(n), ei(n, value_t(0)), evecs(n, n) {}
};

template <typename value_t> dense_matrix<value_t> cross_backend_symmetric_input(std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            // Deterministic, well-separated spectrum (no rand(): every backend sees identical
            // input).
            const value_t v = static_cast<value_t>(std::sin(1.0 + 3.0 * i + 7.0 * j));
            A(i, j)         = v;
            A(j, i)         = v;
        }
        A(i, i) += static_cast<value_t>(i);
    }
    return A;
}

template <typename value_t> dense_matrix<value_t> cross_backend_general_input(std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            A(i, j) = static_cast<value_t>(std::sin(1.0 + 3.0 * i + 7.0 * j));
        }
    }
    return A;
}

template <typename value_t, linalg::linalg_backend Backend>
eigen_result<value_t> run_symmetric(const dense_matrix<value_t>& input, std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    std::copy(input.begin(), input.begin() + n * n, A.begin());
    eigen_result<value_t> r(n);
    auto       ws = symmetric_eigen_workspace<value_t, Backend>(static_cast<linalg_int>(n));
    const bool ok = linalg::symmetric_eigen_backend<value_t, Backend>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        r.evals.data(),
        r.evecs.begin(),
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());
    EXPECT_TRUE(ok);
    return r;
}

template <typename value_t, linalg::linalg_backend Backend>
eigen_result<value_t> run_general(const dense_matrix<value_t>& input, std::size_t n)
{
    dense_matrix<value_t> A(n, n);
    std::copy(input.begin(), input.begin() + n * n, A.begin());
    eigen_result<value_t> r(n);
    auto       ws = general_eigen_workspace<value_t, Backend>(static_cast<linalg_int>(n), false);
    const bool ok = linalg::general_eigen_backend<value_t, Backend>::apply(A.begin(),
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(n),
        r.evals.data(),
        r.ei.data(),
        nullptr,
        static_cast<linalg_int>(n),
        ws.data(),
        ws.size());
    EXPECT_TRUE(ok) << "general eigen failed, n=" << n << " sizeof(T)=" << sizeof(value_t)
                    << " backend=" << static_cast<int>(Backend);
    return r;
}

// Sort eigenvalues (real, imag) lexicographically with imag rounded so that
// conjugate pairs order deterministically (positive imaginary part first).
template <typename value_t>
std::vector<std::pair<value_t, value_t>> sorted_spectrum(const eigen_result<value_t>& r)
{
    std::vector<std::pair<value_t, value_t>> s;
    for (std::size_t i = 0; i < r.evals.size(); i++)
    {
        s.emplace_back(r.evals[i], r.ei[i]);
    }
    const value_t tol = static_cast<value_t>(EigenTolerance<value_t>::tolerance);
    std::sort(s.begin(),
        s.end(),
        [tol](const auto& a, const auto& b)
        {
            if (std::fabs(a.first - b.first) > tol)
            {
                return a.first < b.first;
            }
            return a.second > b.second;
        });
    return s;
}

template <typename value_t>
void compare_spectra(
    const eigen_result<value_t>& ref, const eigen_result<value_t>& got, const char* name)
{
    const auto a = sorted_spectrum(ref);
    const auto b = sorted_spectrum(got);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); i++)
    {
        EXPECT_NEAR(static_cast<double>(a[i].first),
            static_cast<double>(b[i].first),
            EigenTolerance<value_t>::tolerance)
            << name << " eigenvalue real part " << i;
        EXPECT_NEAR(static_cast<double>(a[i].second),
            static_cast<double>(b[i].second),
            EigenTolerance<value_t>::tolerance)
            << name << " eigenvalue imag part " << i;
    }
}

// Eigenvectors are only defined up to sign, so compare |v_ref . v_got| == 1
// per column (columns are unit norm and the spectrum is simple).
template <typename value_t>
void compare_symmetric_vectors(const eigen_result<value_t>& ref,
    const eigen_result<value_t>&                            got,
    std::size_t                                             n,
    const char*                                             name)
{
    for (std::size_t c = 0; c < n; c++)
    {
        double dot = 0;
        for (std::size_t r = 0; r < n; r++)
        {
            dot += static_cast<double>(ref.evecs(r, c)) * static_cast<double>(got.evecs(r, c));
        }
        EXPECT_NEAR(std::fabs(dot), 1.0, EigenTolerance<value_t>::tolerance)
            << name << " eigenvector column " << c;
    }
}

template <typename value_t> void test_cross_backend_agreement(std::size_t n)
{
    const auto sym = cross_backend_symmetric_input<value_t>(n);
    const auto gen = cross_backend_general_input<value_t>(n);

    const auto sym_ref = run_symmetric<value_t, linalg::linalg_backend::native>(sym, n);
    const auto gen_ref = run_general<value_t, linalg::linalg_backend::native>(gen, n);

    linalg_test::for_each_available_backend(
        [&](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            if (Backend == linalg::linalg_backend::native)
            {
                return;
            }
            const char* name = (Backend == linalg::linalg_backend::mkl) ? "mkl" : "blas_lapacke";
            const auto  s    = run_symmetric<value_t, Backend>(sym, n);
            compare_spectra(sym_ref, s, name);
            compare_symmetric_vectors(sym_ref, s, n, name);
            compare_spectra(gen_ref, run_general<value_t, Backend>(gen, n), name);
        });
}
}  // namespace

TEST(Math, SymmetricEigenvalueDecomposition)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            test_symmetric_eigen<float, Backend>(5);
            test_symmetric_eigen<double, Backend>(5);
            test_symmetric_eigen<double, Backend>(8);
        });
}

TEST(Math, GeneralEigenvalueDecompositionRealSpectrum)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            test_general_eigen_real_spectrum<float, Backend>(4);
            test_general_eigen_real_spectrum<double, Backend>(4);
            test_general_eigen_real_spectrum<double, Backend>(6);
        });
}

TEST(Math, GeneralEigenvalueDecompositionComplexPair)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            test_general_eigen_complex_pair<float, Backend>();
            test_general_eigen_complex_pair<double, Backend>();
        });
}

TEST(Math, GeneralEigenvalueDecompositionEigenvectors)
{
    linalg_test::for_each_available_backend(
        [](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;
            test_general_eigen_eigenvectors<float, Backend>();
            test_general_eigen_eigenvectors<double, Backend>();
        });
}

TEST(Math, GeneralEigenvalueDecompositionComplexEigenvectorThrows)
{
    dense_matrix<double> A(2, 2);
    A(0, 0) = 0;
    A(0, 1) = -1;
    A(1, 0) = 1;
    A(1, 1) = 0;

    linalg_test::for_each_available_backend(
        [&](auto backend)
        {
            constexpr auto Backend = decltype(backend)::value;

            double er[2];
            double ei[2];
            double evecs[4];
            auto   eigen_ws = general_eigen_workspace<double, Backend>(2, true);
            // mkl's and blas_lapacke's LAPACKE *geev call packs complex
            // eigenvectors natively (see eigen.h); only the portable scalar
            // fallback, which has no complex arithmetic, throws for this case.
            if (Backend == linalg::linalg_backend::native)
            {
                EXPECT_THROW(
                    (linalg::general_eigen_backend<double, Backend>::apply(
                        A.begin(), 2, 2, er, ei, evecs, 2, eigen_ws.data(), eigen_ws.size())),
                    std::exception);
            }
            else
            {
                EXPECT_NO_THROW((linalg::general_eigen_backend<double, Backend>::apply(
                    A.begin(), 2, 2, er, ei, evecs, 2, eigen_ws.data(), eigen_ws.size())));
            }
        });
}

TEST(Math, EigenBackendsAgree)
{
    test_cross_backend_agreement<float>(5);
    test_cross_backend_agreement<double>(5);
    test_cross_backend_agreement<double>(8);
}
