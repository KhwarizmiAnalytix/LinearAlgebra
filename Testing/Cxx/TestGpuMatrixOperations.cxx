// GPU matrix-operation tests exercising the struct-template API under
// linalg_backend::CUDA_BLAS. This file replaces the former linalg::gpu::*
// free-function tests; the struct API is the only GPU entry point after the
// two-layer architecture was collapsed into a single cublas TU per operation.
// Like all backend-conditional tests in Testing/Cxx (e.g.
// TestCholeskyDecomposition.cxx's #if !LINALG_HAS_MKL guard), this file
// contributes no TEST cases on a scalar-fallback build.
#if LINALG_HAS_CUBLAS

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/condition_number.h"
#include "include/matrix_operation/determinant.h"
#include "include/matrix_operation/eigen.h"
#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/least_squares.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/multiplication_batched.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/pseudo_inverse.h"
#include "include/matrix_operation/qr_decomposition.h"
#include "include/matrix_operation/rank.h"
#include "include/matrix_operation/svd_decomposition.h"
#include "include/matrix_operation/trace.h"
#include "include/matrix_operation/transpose.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg::linalg_backend;
using linalg_test::dense_matrix;

// Short alias used throughout to reduce verbosity.
constexpr linalg_backend CB = linalg_backend::CUDA_BLAS;

template <typename> struct GpuTolerance
{
};

template <> struct GpuTolerance<float>
{
    static constexpr float value = 5.e-3f;
};

template <> struct GpuTolerance<double>
{
    static constexpr double value  = 1.e-9;
    static constexpr double coarse = 1.e-6;
};

// Minimal RAII CUDA device buffer: test-only, mirrors dense_matrix_test_helper's
// scope (not part of the library) — the GPU entry points under test take bare
// device pointers (cudaMalloc'd) and, for scratch, a caller-owned workspace
// buffer (the library itself never calls cudaMalloc — see
// include/common/cuda_workspace.h), so exercising them needs some minimal
// device-memory ownership here, including for that workspace. Allocation
// failures throw so the test reports the CUDA error before using a null pointer.
template <typename T> class device_buffer
{
public:
    explicit device_buffer(std::size_t count) : count_(count)
    {
        if (count_ == 0)
        {
            return;
        }
        const auto status = cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T));
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
        }
    }
    ~device_buffer() { cudaFree(ptr_); }

    device_buffer(const device_buffer&)            = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() { return ptr_; }

    cudaError_t upload(const T* host) const
    {
        return count_ == 0 ? cudaSuccess
                           : cudaMemcpy(ptr_, host, count_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    cudaError_t download(T* host) const
    {
        return count_ == 0 ? cudaSuccess
                           : cudaMemcpy(host, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

private:
    std::size_t count_;
    T*          ptr_ = nullptr;
};

// Explicit test-boundary downloads. These are never part of a library call.
template <typename T, typename Enqueue> T read_gpu_scalar(Enqueue enqueue)
{
    device_buffer<T> result(1);
    enqueue(result.get());
    T host{};
    EXPECT_EQ(result.download(&host), cudaSuccess);
    return host;
}

template <typename T>
bool read_gpu_cholesky(T* A, linalg_int n, linalg::cholesky_decomposition_enum type)
{
    if (!A || n <= 0)
    {
        linalg::cholesky_decomposition<T, CB>::apply(A, n, type, nullptr, 0, nullptr);
        return false;
    }
    const auto bytes = linalg::cholesky_decomposition<T, CB>::workspace_size(n, type);
    device_buffer<unsigned char> workspace(bytes);
    return read_gpu_scalar<int>(
               [&](int* info)
               {
                   linalg::cholesky_decomposition<T, CB>::apply(
                       A, n, type, workspace.get(), bytes, info);
               }) == 0;
}

template <typename T>
bool read_gpu_lu(T* A, linalg_int n, linalg_int*, void* workspace, size_t bytes)
{
    if (!A || n <= 0)
    {
        linalg::lu_decomposition<T, CB>::apply(A, n, nullptr, workspace, bytes, nullptr);
        return false;
    }
    device_buffer<int> pivot(static_cast<size_t>(n));
    return read_gpu_scalar<int>(
               [&](int* info)
               {
                   linalg::lu_decomposition<T, CB>::apply(
                       A, n, pivot.get(), workspace, bytes, info);
               }) == 0;
}

template <typename value_t>
dense_matrix<value_t> random_matrix(
    std::size_t rows, std::size_t cols, std::default_random_engine& gen)
{
    std::uniform_real_distribution<double> distribution(-5., 5.);
    dense_matrix<value_t>                  M(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            M(i, j) = (value_t)distribution(gen);
        }
    }
    return M;
}

template <typename value_t>
value_t max_abs_diff(const dense_matrix<value_t>& a, const dense_matrix<value_t>& b)
{
    value_t max_error = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(a(i, j) - b(i, j))));
        }
    }
    return max_error;
}

// GEMM: C = op(A) * op(B), compared against a naive host reference, for every
// transpose_a/transpose_b combination.
template <typename value_t>
void matrix_multiplication_gpu_test(
    int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    auto a_rows = transpose_a ? depth : rows;
    auto a_cols = transpose_a ? rows : depth;
    auto b_rows = transpose_b ? columns : depth;
    auto b_cols = transpose_b ? depth : columns;

    std::default_random_engine generator;
    auto A = random_matrix<value_t>((std::size_t)a_rows, (std::size_t)a_cols, generator);
    auto B = random_matrix<value_t>((std::size_t)b_rows, (std::size_t)b_cols, generator);
    dense_matrix<value_t> C((std::size_t)rows, (std::size_t)columns);

    device_buffer<value_t> dA((std::size_t)(a_rows * a_cols));
    device_buffer<value_t> dB((std::size_t)(b_rows * b_cols));
    device_buffer<value_t> dC((std::size_t)(rows * columns));
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dB.upload(B.data()), cudaSuccess);

    linalg::multiplication<value_t, CB>::apply(transpose_a,
        transpose_b,
        rows,
        columns,
        depth,
        dA.get(),
        a_cols,
        dB.get(),
        b_cols,
        dC.get(),
        columns,
        nullptr,
        0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(dC.download(C.data()), cudaSuccess);

    dense_matrix<value_t> expected =
        matmul(transpose_a ? transpose(A) : A, transpose_b ? transpose(B) : B);

    EXPECT_LE(max_abs_diff(C, expected), tol);
}

// A (rows x rows), positive definite, built as R^T * R from a random R with
// a strictly positive diagonal, so it is well-conditioned for both the
// Cholesky and LU paths below.
template <typename value_t>
dense_matrix<value_t> spd_matrix(std::size_t n, std::default_random_engine& gen)
{
    auto R = random_matrix<value_t>(n, n, gen);
    for (std::size_t i = 0; i < n; ++i)
    {
        R(i, i) += (value_t)(2 * n);  // diagonally dominant -> well-conditioned
    }
    return matmul(transpose(R), R);
}

template <typename value_t> void linear_solver_gpu_test(linalg::linear_solver_type type)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const int      n   = 37;

    std::default_random_engine generator;
    auto                       A      = spd_matrix<value_t>((std::size_t)n, generator);
    auto                       x_true = random_matrix<value_t>((std::size_t)n, 1, generator);
    auto                       b      = matmul(A, x_true);

    device_buffer<value_t> dA((std::size_t)(n * n));
    device_buffer<value_t> dx((std::size_t)n);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dx.upload(b.data()), cudaSuccess);

    const auto ws_size =
        linalg::linear_solver<value_t, CB>::workspace_size((linalg_int)n, type);
    device_buffer<unsigned char> dws(ws_size);

    linalg::linear_solver<value_t, CB>::apply(dA.get(), nullptr, (linalg_int)n, dx.get(), type,
        dws.get(), ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<value_t> x_solved((std::size_t)n);
    ASSERT_EQ(dx.download(x_solved.data()), cudaSuccess);

    value_t max_error = 0;
    for (int i = 0; i < n; ++i)
    {
        max_error = std::fmax(max_error,
            std::fabs(static_cast<double>(x_solved[(std::size_t)i] - x_true((std::size_t)i, 0))));
    }
    EXPECT_LE(max_error, tol * n);
}

}  // namespace

TEST(MathGpu, DeviceMemoryAllocation)
{
    device_buffer<double> data(4);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, data.get()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    const auto bytes = linalg::transpose<double, CB>::workspace_size(2, 2);
    ASSERT_GT(bytes, 0U);
    device_buffer<unsigned char> workspace(bytes);
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, workspace.get()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    const double input[] = {1., 2., 3., 4.};
    double       output[4]{};
    ASSERT_EQ(data.upload(input), cudaSuccess);
    linalg::transpose<double, CB>::apply(2, 2, data.get(), workspace.get(), bytes);
    ASSERT_EQ(data.download(output), cudaSuccess);
    EXPECT_EQ(output[0], 1.);
    EXPECT_EQ(output[1], 3.);
    EXPECT_EQ(output[2], 2.);
    EXPECT_EQ(output[3], 4.);

    device_buffer<unsigned char> empty_workspace(0);
    EXPECT_EQ(empty_workspace.get(), nullptr);
    EXPECT_EQ(empty_workspace.upload(nullptr), cudaSuccess);
    EXPECT_EQ(empty_workspace.download(nullptr), cudaSuccess);
}

TEST(MathGpu, MatrixMultiplication)
{
    const int rows = 37, columns = 53, depth = 29;

    matrix_multiplication_gpu_test<float>(rows, columns, depth, false, false);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, true, false);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, false, true);
    matrix_multiplication_gpu_test<float>(rows, columns, depth, true, true);

    matrix_multiplication_gpu_test<double>(rows, columns, depth, false, false);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, true, false);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, false, true);
    matrix_multiplication_gpu_test<double>(rows, columns, depth, true, true);
}

TEST(MathGpu, MatrixTranspose)
{
    const std::size_t rows = 41, columns = 23;

    std::default_random_engine generator;
    auto                       A_f = random_matrix<float>(rows, columns, generator);
    auto                       A_d = random_matrix<double>(rows, columns, generator);

    device_buffer<float> dA_f(rows * columns);
    ASSERT_EQ(dA_f.upload(A_f.data()), cudaSuccess);
    const auto ws_size_f =
        linalg::transpose<float, CB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws_f(ws_size_f);
    linalg::transpose<float, CB>::apply(
        (linalg_long)rows, (linalg_long)columns, dA_f.get(), dws_f.get(), ws_size_f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dense_matrix<float> T_f(columns, rows);
    ASSERT_EQ(dA_f.download(T_f.data()), cudaSuccess);
    EXPECT_LE(max_abs_diff(T_f, transpose(A_f)), GpuTolerance<float>::value);

    device_buffer<double> dA_d(rows * columns);
    ASSERT_EQ(dA_d.upload(A_d.data()), cudaSuccess);
    const auto ws_size_d =
        linalg::transpose<double, CB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws_d(ws_size_d);
    linalg::transpose<double, CB>::apply(
        (linalg_long)rows, (linalg_long)columns, dA_d.get(), dws_d.get(), ws_size_d);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dense_matrix<double> T_d(columns, rows);
    ASSERT_EQ(dA_d.download(T_d.data()), cudaSuccess);
    EXPECT_LE(max_abs_diff(T_d, transpose(A_d)), GpuTolerance<double>::value);
}

TEST(MathGpu, BatchedMatrixMultiplication)
{
    const int dim = 9, count = 6;

    std::default_random_engine             generator;
    std::vector<double>                    a((std::size_t)(dim * dim * count));
    std::vector<double>                    b((std::size_t)(dim * dim * count));
    std::vector<double>                    c((std::size_t)(dim * dim * count), 0.);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (auto& v : a)
        v = distribution(generator);
    for (auto& v : b)
        v = distribution(generator);

    device_buffer<double> da(a.size());
    device_buffer<double> db(b.size());
    device_buffer<double> dc(c.size());
    ASSERT_EQ(da.upload(a.data()), cudaSuccess);
    ASSERT_EQ(db.upload(b.data()), cudaSuccess);

    linalg::batched_matrix_multiplication(dim, count, da.get(), db.get(), dc.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(dc.download(c.data()), cudaSuccess);

    double max_error = 0;
    for (int mat = 0; mat < count; ++mat)
    {
        const double* Ai = a.data() + (std::size_t)(mat * dim * dim);
        const double* Bi = b.data() + (std::size_t)(mat * dim * dim);
        const double* Ci = c.data() + (std::size_t)(mat * dim * dim);
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                double sum = 0.;
                for (int k = 0; k < dim; ++k)
                {
                    sum += Ai[i * dim + k] * Bi[k * dim + j];
                }
                max_error = std::fmax(max_error, std::fabs(sum - Ci[i * dim + j]));
            }
        }
    }
    EXPECT_LE(max_error, GpuTolerance<double>::value * dim);
}

TEST(MathGpu, CholeskyDecomposition)
{
    constexpr auto    tol = GpuTolerance<double>::value;
    const std::size_t n   = 31;

    std::default_random_engine generator;
    // Known lower-triangular factor with a positive diagonal.
    dense_matrix<double>                   L(n, n);
    std::uniform_real_distribution<double> distribution(-5., 5.);
    for (std::size_t i = 0; i < n; ++i)
    {
        L(i, i) = distribution(generator) + 10.;
        for (std::size_t j = 0; j < i; ++j)
        {
            L(i, j) = distribution(generator);
        }
    }
    dense_matrix<double> A = matmul(L, transpose(L));

    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    // Observe device status only at the test boundary.
    const bool ok = read_gpu_cholesky<double>(
        dA.get(), (linalg_int)n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(ok);

    dense_matrix<double> R(n, n);
    ASSERT_EQ(dA.download(R.data()), cudaSuccess);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            R(i, j) = 0.;
        }
    }

    EXPECT_LE(max_abs_diff(R, L), tol * (double)n);
}

TEST(MathGpu, LinearSolverCholesky)
{
    linear_solver_gpu_test<float>(linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
    linear_solver_gpu_test<double>(linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
}

TEST(MathGpu, LinearSolverLU)
{
    linear_solver_gpu_test<float>(linalg::linear_solver_type::LU_LINEAR_SOLVER);
    linear_solver_gpu_test<double>(linalg::linear_solver_type::LU_LINEAR_SOLVER);
}

// Verify packed LU and the caller-owned device pivot sequence.
TEST(MathGpu, LuDecomposition)
{
    constexpr auto                         tol = GpuTolerance<double>::value;
    const std::size_t                      n   = 17;
    std::default_random_engine             generator;
    auto                                   A = random_matrix<double>(n, n, generator);

    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    const auto ws_size =
        linalg::lu_decomposition<double, CB>::workspace_size((linalg_int)n);
    device_buffer<unsigned char> dws(ws_size);

    device_buffer<int> dpivot(n);
    device_buffer<int> dinfo(1);
    linalg::lu_decomposition<double, CB>::apply(
        dA.get(), (linalg_int)n, dpivot.get(), dws.get(), ws_size, dinfo.get());
    int info = -1;
    ASSERT_EQ(dinfo.download(&info), cudaSuccess);
    const bool ok = info == 0;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(ok);

    dense_matrix<double> packed(n, n);
    ASSERT_EQ(dA.download(packed.data()), cudaSuccess);

    // Unpack L (unit lower triangular) and U (upper triangular) from the
    // row-major packed factor, exactly as TestLUDecomposition.cxx does for
    // the CPU path.
    dense_matrix<double> Lmat(n, n);
    dense_matrix<double> U(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        U(i, i)    = packed(i, i);
        Lmat(i, i) = 1.;
        for (std::size_t j = 0; j < i; ++j)
        {
            Lmat(i, j) = packed(i, j);
            U(j, i)    = packed(j, i);
        }
    }

    std::vector<int> pivot(n);
    ASSERT_EQ(dpivot.download(pivot.data()), cudaSuccess);
    auto PA = A;
    for (std::size_t i = 0; i < n; ++i)
    {
        ASSERT_GE(pivot[i], 1);
        ASSERT_LE(pivot[i], static_cast<int>(n));
        for (std::size_t j = 0; j < n; ++j)
            std::swap(PA(i, j), PA(static_cast<std::size_t>(pivot[i] - 1), j));
    }
    EXPECT_LE(max_abs_diff(matmul(Lmat, U), PA), tol * (double)n);
}

// A * A^-1 == I for both LU- and Cholesky-based inversion on the same SPD
// matrix (valid input for either factorization), and the two backends'
// determinants agree with each other — mirrors TestMatrixInversion.cxx's
// CPU coverage, adapted to the in-place struct API.
TEST(MathGpu, MatrixInversion)
{
    constexpr auto    tol = GpuTolerance<double>::value;
    const std::size_t n   = 23;

    std::default_random_engine generator;
    auto                       A = spd_matrix<double>(n, generator);

    dense_matrix<double> Id(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        Id(i, i) = 1.;
    }

    for (auto type : {linalg::linear_solver_type::LU_LINEAR_SOLVER,
             linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER})
    {
        device_buffer<double> dA(n * n);
        ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

        const auto ws_size =
            linalg::inversion<double, CB>::workspace_size((linalg_int)n, type);
        device_buffer<unsigned char> dws(ws_size);

        // nullptr pivot: ignored for cuda_blas.
        linalg::inversion<double, CB>::apply(
            dA.get(), nullptr, (linalg_int)n, type, dws.get(), ws_size);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        dense_matrix<double> inv(n, n);
        ASSERT_EQ(dA.download(inv.data()), cudaSuccess);

        auto product = matmul(inv, A);
        EXPECT_LE(max_abs_diff(product, Id), tol * (double)n)
            << "type=" << static_cast<long long>(type);
    }

    device_buffer<double> dA_lu(n * n);
    device_buffer<double> dA_chol(n * n);
    ASSERT_EQ(dA_lu.upload(A.data()), cudaSuccess);
    ASSERT_EQ(dA_chol.upload(A.data()), cudaSuccess);

    const auto ws_size_lu = linalg::determinant<double, CB>::workspace_size(
        (linalg_int)n, linalg::linear_solver_type::LU_LINEAR_SOLVER);
    device_buffer<unsigned char> dws_lu(ws_size_lu);
    const auto ws_size_chol = linalg::determinant<double, CB>::workspace_size(
        (linalg_int)n, linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
    device_buffer<unsigned char> dws_chol(ws_size_chol);

    const double det_lu = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::determinant<double, CB>::apply(dA_lu.get(),
                nullptr,
                (linalg_int)n,
                linalg::linear_solver_type::LU_LINEAR_SOLVER,
                dws_lu.get(),
                ws_size_lu,
                result);
        });
    const double det_chol = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::determinant<double, CB>::apply(dA_chol.get(),
                nullptr,
                (linalg_int)n,
                linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER,
                dws_chol.get(),
                ws_size_chol,
                result);
        });

    EXPECT_GT(det_lu, 0.);  // A is SPD
    EXPECT_LE(std::fabs(det_lu - det_chol), tol * std::fabs(det_lu));
}

// A == U * diag(S) * VT reconstruction, mirroring TestSVDDecomposition.cxx's
// CPU test_svd<T> exactly (same ldu = min(rows, columns), ldv = columns
// packing the CPU test already uses), for both a tall and a wide matrix.
template <typename value_t> void svd_decomposition_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const auto     k   = std::min(rows, columns);

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dS(k);
    device_buffer<value_t> dU(rows * k);
    device_buffer<value_t> dVT(k * columns);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    const auto ws_size =
        linalg::svd_decomposition<value_t, CB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws(ws_size);

    linalg::svd_decomposition<value_t, CB>::apply((linalg_long)rows,
        (linalg_long)columns,
        dA.get(),
        (linalg_long)columns,
        dS.get(),
        dU.get(),
        (linalg_long)k,
        dVT.get(),
        (linalg_long)columns,
        dws.get(),
        ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<value_t>  S(k);
    dense_matrix<value_t> U(rows, k);
    dense_matrix<value_t> VT(k, columns);
    ASSERT_EQ(dS.download(S.data()), cudaSuccess);
    ASSERT_EQ(dU.download(U.data()), cudaSuccess);
    ASSERT_EQ(dVT.download(VT.data()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = A(i, j);
            for (std::size_t kk = 0; kk < k; ++kk)
            {
                sum -= U(i, kk) * S[kk] * VT(kk, j);
            }
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(sum)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)k);
}

TEST(MathGpu, SVDDecomposition)
{
    svd_decomposition_gpu_test<float>(23, 11);
    svd_decomposition_gpu_test<double>(23, 11);

    svd_decomposition_gpu_test<float>(11, 23);
    svd_decomposition_gpu_test<double>(11, 23);
}

// A == Q*R reconstruction plus Q^T*Q == I, mirroring TestQRDecomposition.cxx's
// CPU test_qr<T> and svd_decomposition_gpu_test's device-buffer plumbing.
template <typename value_t> void qr_decomposition_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;
    const auto     k   = std::min(rows, columns);

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dtau(k);
    ASSERT_EQ(dA.upload(A.data()), cudaSuccess);

    const auto ws_size = linalg::qr_decomposition<value_t, CB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws(ws_size);

    linalg::qr_decomposition<value_t, CB>::apply_r(
        rows, columns, dA.get(), columns, dtau.get(), dws.get(), ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Saving R is this client's decision; the factorization itself leaves it packed.
    dense_matrix<value_t> packed(rows, columns);
    ASSERT_EQ(dA.download(packed.begin()), cudaSuccess);
    dense_matrix<value_t> R(k, columns);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = i; j < columns; ++j)
            R(i, j) = packed(i, j);
    linalg::qr_decomposition<value_t, CB>::apply_q(
        rows, columns, dA.get(), columns, dtau.get(), dws.get(), ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dense_matrix<value_t> Q(rows, columns);
    ASSERT_EQ(dA.download(Q.begin()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = 0;
            for (std::size_t c = 0; c < k; ++c)
            {
                sum += Q(i, c) * R(c, j);
            }
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(A(i, j) - sum)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)k);

    value_t max_orth_error = 0;
    for (std::size_t i = 0; i < k; ++i)
    {
        for (std::size_t j = 0; j < k; ++j)
        {
            value_t sum = 0;
            for (std::size_t r = 0; r < rows; ++r)
            {
                sum += Q(r, i) * Q(r, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_orth_error =
                std::fmax(max_orth_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_orth_error, tol * (value_t)k);
}

TEST(MathGpu, QRDecomposition)
{
    qr_decomposition_gpu_test<float>(23, 11);
    qr_decomposition_gpu_test<double>(23, 11);

    qr_decomposition_gpu_test<float>(11, 23);
    qr_decomposition_gpu_test<double>(11, 23);
}

// A = Q * diag(known) * Q^T for a random orthogonal Q built from the GPU QR
// above; eigenvalues must come back ascending and reconstruct A.
template <typename value_t> void symmetric_eigen_gpu_test(std::size_t n)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    std::default_random_engine generator;
    auto                       R0 = random_matrix<value_t>(n, n, generator);

    device_buffer<value_t> dR0(n * n);
    device_buffer<value_t> dtau(n);
    ASSERT_EQ(dR0.upload(R0.begin()), cudaSuccess);
    const auto qr_ws_size =
        linalg::qr_decomposition<value_t, CB>::workspace_size((linalg_long)n, (linalg_long)n);
    device_buffer<unsigned char> qr_dws(qr_ws_size);
    linalg::qr_decomposition<value_t, CB>::apply_r((linalg_long)n,
        (linalg_long)n,
        dR0.get(),
        (linalg_long)n,
        dtau.get(),
        qr_dws.get(),
        qr_ws_size);
    linalg::qr_decomposition<value_t, CB>::apply_q((linalg_long)n,
        (linalg_long)n,
        dR0.get(),
        (linalg_long)n,
        dtau.get(),
        qr_dws.get(),
        qr_ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<value_t> Q(n, n);
    ASSERT_EQ(dR0.download(Q.begin()), cudaSuccess);

    std::vector<value_t> known(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        known[i] = (value_t)(i + 1);
    }

    dense_matrix<value_t> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += Q(i, k) * known[k] * Q(j, k);
            }
            A(i, j) = sum;
        }
    }

    device_buffer<value_t> dA(n * n);
    device_buffer<value_t> devals(n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    const auto eig_ws_size =
        linalg::symmetric_eigen_backend<value_t, CB>::workspace_size((linalg_int)n);
    device_buffer<unsigned char> eig_dws(eig_ws_size);

    const bool ok = read_gpu_scalar<int>(
                        [&](int* info)
                        {
                            linalg::symmetric_eigen_backend<value_t, CB>::apply(dA.get(),
                                (linalg_int)n,
                                (linalg_int)n,
                                devals.get(),
                                (linalg_int)n,
                                eig_dws.get(),
                                eig_ws_size,
                                info);
                        }) == 0;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(ok);

    std::vector<value_t>  evals(n);
    dense_matrix<value_t> evecs(n, n);
    ASSERT_EQ(devals.download(evals.data()), cudaSuccess);
    ASSERT_EQ(dA.download(evecs.begin()), cudaSuccess);

    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR((double)evals[i], (double)known[i], tol * (double)n);
    }

    // AV = A * evecs, then evecs^T * AV should reconstruct diag(evals) (two
    // matmul passes rather than one quad-nested loop, matching this
    // repository's CPU TestEigenvalueDecomposition.cxx workaround for a
    // clang-cl optimizer crash on that pattern).
    dense_matrix<value_t> AV(n, n);
    for (std::size_t p = 0; p < n; ++p)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t q = 0; q < n; ++q)
            {
                sum += A(p, q) * evecs(q, j);
            }
            AV(p, j) = sum;
        }
    }
    value_t max_error = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            value_t sum = 0;
            for (std::size_t p = 0; p < n; ++p)
            {
                sum += evecs(p, i) * AV(p, j);
            }
            const value_t expected = (i == j) ? evals[i] : value_t(0);
            max_error =
                std::fmax(max_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)n);
}

TEST(MathGpu, SymmetricEigenvalueDecomposition)
{
    symmetric_eigen_gpu_test<float>(6);
    symmetric_eigen_gpu_test<double>(6);
}

// Full column-rank tall matrix: pinv(A) is a left inverse.
template <typename value_t> void pseudo_inverse_gpu_test(std::size_t rows, std::size_t columns)
{
    constexpr auto tol = GpuTolerance<value_t>::value;

    std::default_random_engine generator;
    auto                       A = random_matrix<value_t>(rows, columns, generator);

    device_buffer<value_t> dA(rows * columns);
    device_buffer<value_t> dAinv(columns * rows);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    const auto ws_size =
        linalg::pseudo_inverse<value_t, CB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws(ws_size);

    linalg::pseudo_inverse<value_t, CB>::apply((linalg_long)rows, (linalg_long)columns, dA.get(),
        (linalg_long)columns, dAinv.get(), (linalg_long)rows, static_cast<value_t>(-1), dws.get(),
        ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<value_t> Ainv(columns, rows);
    ASSERT_EQ(dAinv.download(Ainv.begin()), cudaSuccess);

    value_t max_error = 0;
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            value_t sum = 0;
            for (std::size_t k = 0; k < rows; ++k)
            {
                sum += Ainv(i, k) * A(k, j);
            }
            const value_t expected = (i == j) ? value_t(1) : value_t(0);
            max_error = std::fmax(max_error, std::fabs(static_cast<double>(sum - expected)));
        }
    }
    EXPECT_LE(max_error, tol * (value_t)rows);
}

TEST(MathGpu, PseudoInverse)
{
    pseudo_inverse_gpu_test<float>(9, 5);
    pseudo_inverse_gpu_test<double>(9, 5);
}

TEST(MathGpu, MatrixRankAndConditionNumber)
{
    const std::size_t n = 6;
    dense_matrix<double> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        I(i, i) = 1.0;
    }
    device_buffer<double> dI(n * n);
    ASSERT_EQ(dI.upload(I.begin()), cudaSuccess);

    const auto ws_size =
        linalg::rank<double, CB>::workspace_size((linalg_long)n, (linalg_long)n);
    device_buffer<unsigned char> dws(ws_size);

    const auto r = read_gpu_scalar<linalg_long>(
        [&](linalg_long* result)
        {
            linalg::rank<double, CB>::apply(dI.get(),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                -1.,
                dws.get(),
                ws_size,
                result);
        });
    const auto cond = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::condition_number<double, CB>::apply(dI.get(),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                dws.get(),
                ws_size,
                result);
        });
    EXPECT_EQ(r, (linalg_long)n);
    EXPECT_NEAR(cond, 1.0, GpuTolerance<double>::coarse);

    // Rank-1 outer product.
    std::default_random_engine generator;
    std::uniform_real_distribution<double> dist(1., 2.);
    dense_matrix<double>                   A(n, n);
    std::vector<double>                    u(n), v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        u[i] = dist(generator);
        v[i] = dist(generator);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            A(i, j) = u[i] * v[j];
        }
    }
    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    // Reuse same workspace size (rank of the rank-1 matrix is also n x n).
    const auto rank1 = read_gpu_scalar<linalg_long>(
        [&](linalg_long* result)
        {
            linalg::rank<double, CB>::apply(dA.get(),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                -1.,
                dws.get(),
                ws_size,
                result);
        });
    EXPECT_EQ(rank1, (linalg_long)1);
}

TEST(MathGpu, MatrixNorm)
{
    const std::size_t rows = 6, columns = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix<double>(rows, columns, generator);

    device_buffer<double> dA(rows * columns);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    double frob_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            frob_expected += A(i, j) * A(i, j);
        }
    }
    frob_expected = std::sqrt(frob_expected);

    double one_expected = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        double s = 0;
        for (std::size_t i = 0; i < rows; ++i)
        {
            s += std::fabs(A(i, j));
        }
        one_expected = std::max(one_expected, s);
    }

    double inf_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        double s = 0;
        for (std::size_t j = 0; j < columns; ++j)
        {
            s += std::fabs(A(i, j));
        }
        inf_expected = std::max(inf_expected, s);
    }

    const auto frob_ws_size = linalg::norm<double, CB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::FROBENIUS);
    device_buffer<unsigned char> frob_dws(frob_ws_size);
    const auto one_ws_size = linalg::norm<double, CB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::ONE);
    device_buffer<unsigned char> one_dws(one_ws_size);
    const auto inf_ws_size = linalg::norm<double, CB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::INFINITY_NORM);
    device_buffer<unsigned char> inf_dws(inf_ws_size);

    const auto frob = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::norm<double, CB>::apply(dA.get(),
                (linalg_long)rows,
                (linalg_long)columns,
                (linalg_long)columns,
                linalg::norm_type::FROBENIUS,
                frob_dws.get(),
                frob_ws_size,
                result);
        });
    const auto one = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::norm<double, CB>::apply(dA.get(),
                (linalg_long)rows,
                (linalg_long)columns,
                (linalg_long)columns,
                linalg::norm_type::ONE,
                one_dws.get(),
                one_ws_size,
                result);
        });
    const auto inf_norm = read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::norm<double, CB>::apply(dA.get(),
                (linalg_long)rows,
                (linalg_long)columns,
                (linalg_long)columns,
                linalg::norm_type::INFINITY_NORM,
                inf_dws.get(),
                inf_ws_size,
                result);
        });

    EXPECT_NEAR(frob, frob_expected, GpuTolerance<double>::coarse);
    EXPECT_NEAR(one, one_expected, GpuTolerance<double>::coarse);
    EXPECT_NEAR(inf_norm, inf_expected, GpuTolerance<double>::coarse);
}

TEST(MathGpu, MatrixTrace)
{
    const std::size_t n = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix<double>(n, n, generator);

    double expected = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        expected += A(i, i);
    }

    device_buffer<double> dA(n * n);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);

    // Observe the device scalar only at the test boundary.
    const auto tr = read_gpu_scalar<double>([&](double* result)
        { linalg::trace<double, CB>::apply(dA.get(), (linalg_int)n, (linalg_int)n, result); });
    EXPECT_NEAR(tr, expected, GpuTolerance<double>::value);
}

// Overdetermined, consistent system: the least-squares solution should
// recover the exact generating X (residual is zero).
TEST(MathGpu, LeastSquares)
{
    const std::size_t rows = 6, columns = 3, nrhs = 2;

    dense_matrix<double> A(rows, columns);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < columns; ++j)
        {
            A(i, j) = (double)(1 + (i * 3 + j * 7) % 11);
        }
    }
    dense_matrix<double> X_true(columns, nrhs);
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            X_true(i, j) = (double)(1 + i + 2 * j);
        }
    }
    dense_matrix<double> B(rows, nrhs);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            double sum = 0;
            for (std::size_t k = 0; k < columns; ++k)
            {
                sum += A(i, k) * X_true(k, j);
            }
            B(i, j) = sum;
        }
    }

    device_buffer<double> dA(rows * columns);
    device_buffer<double> dB(rows * nrhs);
    device_buffer<double> dX(columns * nrhs);
    ASSERT_EQ(dA.upload(A.begin()), cudaSuccess);
    ASSERT_EQ(dB.upload(B.begin()), cudaSuccess);

    const auto ws_size =
        linalg::least_squares<double, CB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    device_buffer<unsigned char> dws(ws_size);

    linalg::least_squares<double, CB>::apply((linalg_long)rows,
        (linalg_long)columns,
        (linalg_long)nrhs,
        dA.get(),
        (linalg_long)columns,
        dB.get(),
        (linalg_long)nrhs,
        dX.get(),
        (linalg_long)nrhs,
        dws.get(),
        ws_size);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    dense_matrix<double> X(columns, nrhs);
    ASSERT_EQ(dX.download(X.begin()), cudaSuccess);

    double max_error = 0;
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            max_error = std::fmax(max_error, std::fabs(X(i, j) - X_true(i, j)));
        }
    }
    EXPECT_LT(max_error, GpuTolerance<double>::coarse);
}

// Every struct-API entry point throws (rather than exhibiting undefined
// behavior) on a null required device pointer or a non-positive dimension —
// mirrors Math.InputValidation's CPU coverage in TestInputValidation.cxx.
// One representative case per op is enough here; this is a regression guard
// for that contract, not an exhaustive sweep.
TEST(MathGpu, InputValidation)
{
    const int n = 3;

    device_buffer<double> dA((std::size_t)(n * n));
    device_buffer<double> dx((std::size_t)n);
    device_buffer<double> dS((std::size_t)n);

    // cholesky_decomposition — device workspace and numerical status.
    EXPECT_ANY_THROW((read_gpu_cholesky<double>(
        static_cast<double*>(nullptr), n, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)));
    EXPECT_ANY_THROW((read_gpu_cholesky<double>(
        dA.get(), 0, linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR)));

    // lu_decomposition.
    EXPECT_ANY_THROW((read_gpu_lu<double>(static_cast<double*>(nullptr), n, nullptr, nullptr, 0)));
    EXPECT_ANY_THROW((read_gpu_lu<double>(dA.get(), 0, nullptr, nullptr, 0)));

    // linear_solver.
    EXPECT_ANY_THROW((linalg::linear_solver<double, CB>::apply(static_cast<double*>(nullptr),
        nullptr,
        n,
        dx.get(),
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::linear_solver<double, CB>::apply(
        dA.get(), nullptr, 0, dx.get(), linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));

    // inversion.
    EXPECT_ANY_THROW((linalg::inversion<double, CB>::apply(static_cast<double*>(nullptr),
        nullptr,
        n,
        linalg::linear_solver_type::LU_LINEAR_SOLVER,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::inversion<double, CB>::apply(
        dA.get(), nullptr, 0, linalg::linear_solver_type::LU_LINEAR_SOLVER, nullptr, 0)));

    // determinant.
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::determinant<double, CB>::apply(static_cast<double*>(nullptr),
                nullptr,
                n,
                linalg::linear_solver_type::LU_LINEAR_SOLVER,
                nullptr,
                0,
                result);
        })));

    // multiplication.
    EXPECT_ANY_THROW((linalg::multiplication<double, CB>::apply(false,
        false,
        n,
        n,
        n,
        static_cast<const double*>(nullptr),
        n,
        dA.get(),
        n,
        dA.get(),
        n,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::multiplication<double, CB>::apply(
        false, false, 0, n, n, dA.get(), n, dA.get(), n, dA.get(), n, nullptr, 0)));

    // transpose.
    EXPECT_ANY_THROW((linalg::transpose<double, CB>::apply(
        (linalg_long)n, (linalg_long)n, static_cast<double*>(nullptr), nullptr, 0)));
    EXPECT_ANY_THROW(
        (linalg::transpose<double, CB>::apply(0LL, (linalg_long)n, dA.get(), nullptr, 0)));

    // svd_decomposition.
    EXPECT_ANY_THROW((linalg::svd_decomposition<double, CB>::apply((linalg_long)n,
        (linalg_long)n,
        static_cast<double*>(nullptr),
        (linalg_long)n,
        dS.get(),
        dA.get(),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::svd_decomposition<double, CB>::apply(0LL,
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        dS.get(),
        dA.get(),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        nullptr,
        0)));

    // qr_decomposition.
    EXPECT_ANY_THROW((linalg::qr_decomposition<double, CB>::apply_r(
        n, n, static_cast<double*>(nullptr), n, dA.get(), nullptr, 0)));
    EXPECT_ANY_THROW(
        (linalg::qr_decomposition<double, CB>::apply_r(0, n, dA.get(), n, dA.get(), nullptr, 0)));

    // symmetric_eigen_backend.
    EXPECT_ANY_THROW((read_gpu_scalar<int>(
        [&](int* info)
        {
            linalg::symmetric_eigen_backend<double, CB>::apply(
                static_cast<double*>(nullptr), n, n, dS.get(), n, nullptr, 0, info);
        })));
    EXPECT_ANY_THROW((read_gpu_scalar<int>(
        [&](int* info)
        {
            linalg::symmetric_eigen_backend<double, CB>::apply(
                dA.get(), 0, 0, dS.get(), 0, nullptr, 0, info);
        })));

    // pseudo_inverse.
    EXPECT_ANY_THROW((linalg::pseudo_inverse<double, CB>::apply((linalg_long)n,
        (linalg_long)n,
        static_cast<const double*>(nullptr),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        -1.,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::pseudo_inverse<double, CB>::apply(
        0LL, (linalg_long)n, dA.get(), (linalg_long)n, dA.get(), (linalg_long)n, -1., nullptr, 0)));

    // rank.
    EXPECT_ANY_THROW((read_gpu_scalar<linalg_long>(
        [&](linalg_long* result)
        {
            linalg::rank<double, CB>::apply(static_cast<const double*>(nullptr),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                -1.,
                nullptr,
                0,
                result);
        })));
    EXPECT_ANY_THROW((read_gpu_scalar<linalg_long>(
        [&](linalg_long* result)
        {
            linalg::rank<double, CB>::apply(
                dA.get(), 0LL, (linalg_long)n, (linalg_long)n, -1., nullptr, 0, result);
        })));

    // condition_number.
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::condition_number<double, CB>::apply(static_cast<const double*>(nullptr),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                nullptr,
                0,
                result);
        })));
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::condition_number<double, CB>::apply(
                dA.get(), 0LL, (linalg_long)n, (linalg_long)n, nullptr, 0, result);
        })));

    // norm.
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::norm<double, CB>::apply(static_cast<const double*>(nullptr),
                (linalg_long)n,
                (linalg_long)n,
                (linalg_long)n,
                linalg::norm_type::FROBENIUS,
                nullptr,
                0,
                result);
        })));
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result)
        {
            linalg::norm<double, CB>::apply(dA.get(),
                0LL,
                (linalg_long)n,
                (linalg_long)n,
                linalg::norm_type::FROBENIUS,
                nullptr,
                0,
                result);
        })));

    // trace — no workspace parameter.
    EXPECT_ANY_THROW((read_gpu_scalar<double>([&](double* result)
        { linalg::trace<double, CB>::apply(static_cast<const double*>(nullptr), n, n, result); })));
    EXPECT_ANY_THROW((read_gpu_scalar<double>(
        [&](double* result) { linalg::trace<double, CB>::apply(dA.get(), 0, n, result); })));

    // least_squares.
    EXPECT_ANY_THROW((linalg::least_squares<double, CB>::apply((linalg_long)n,
        (linalg_long)n,
        (linalg_long)n,
        static_cast<const double*>(nullptr),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        nullptr,
        0)));
    EXPECT_ANY_THROW((linalg::least_squares<double, CB>::apply(0LL,
        (linalg_long)n,
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        dA.get(),
        (linalg_long)n,
        nullptr,
        0)));
}

// Two independent Cholesky solves issued on two distinct caller-created
// streams. Does not attempt to assert actual concurrency/overlap (timing-
// sensitive, unsuited to a correctness test) — the goal is that per-device
// handle reuse (include/common/cuda_handle.h) and the stream parameter
// threaded through every cuda_blas struct entry point do not deadlock or
// corrupt either solve's result.
TEST(MathGpu, StreamParameter)
{
    constexpr auto tol = GpuTolerance<double>::value;
    const int      n   = 23;

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);

    std::default_random_engine generator;
    auto                       A_a      = spd_matrix<double>((std::size_t)n, generator);
    auto                       A_b      = spd_matrix<double>((std::size_t)n, generator);
    auto                       x_true_a = random_matrix<double>((std::size_t)n, 1, generator);
    auto                       x_true_b = random_matrix<double>((std::size_t)n, 1, generator);
    auto                       b_a      = matmul(A_a, x_true_a);
    auto                       b_b      = matmul(A_b, x_true_b);

    device_buffer<double> dA_a((std::size_t)(n * n));
    device_buffer<double> dx_a((std::size_t)n);
    device_buffer<double> dA_b((std::size_t)(n * n));
    device_buffer<double> dx_b((std::size_t)n);
    ASSERT_EQ(dA_a.upload(A_a.data()), cudaSuccess);
    ASSERT_EQ(dx_a.upload(b_a.data()), cudaSuccess);
    ASSERT_EQ(dA_b.upload(A_b.data()), cudaSuccess);
    ASSERT_EQ(dx_b.upload(b_b.data()), cudaSuccess);

    const auto ws_size = linalg::linear_solver<double, CB>::workspace_size(
        (linalg_int)n, linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER);
    device_buffer<unsigned char> dws_a(ws_size);
    device_buffer<unsigned char> dws_b(ws_size);

    linalg::linear_solver<double, CB>::apply(dA_a.get(), nullptr, (linalg_int)n, dx_a.get(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER, dws_a.get(), ws_size,
        static_cast<void*>(stream_a));
    linalg::linear_solver<double, CB>::apply(dA_b.get(), nullptr, (linalg_int)n, dx_b.get(),
        linalg::linear_solver_type::CHOLESKY_LINEAR_SOLVER, dws_b.get(), ws_size,
        static_cast<void*>(stream_b));

    ASSERT_EQ(cudaStreamSynchronize(stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_b), cudaSuccess);

    std::vector<double> x_solved_a((std::size_t)n);
    std::vector<double> x_solved_b((std::size_t)n);
    ASSERT_EQ(dx_a.download(x_solved_a.data()), cudaSuccess);
    ASSERT_EQ(dx_b.download(x_solved_b.data()), cudaSuccess);

    double max_error = 0;
    for (int i = 0; i < n; ++i)
    {
        max_error = std::fmax(
            max_error, std::fabs(x_solved_a[(std::size_t)i] - x_true_a((std::size_t)i, 0)));
        max_error = std::fmax(
            max_error, std::fabs(x_solved_b[(std::size_t)i] - x_true_b((std::size_t)i, 0)));
    }
    EXPECT_LE(max_error, tol * n);

    ASSERT_EQ(cudaStreamDestroy(stream_a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream_b), cudaSuccess);
}

// Feed device scalars into subsequent GPU calls on a caller stream. The
// Frobenius norm changes cuBLAS pointer mode, so the GEMM also checks restoration.
template <typename T> void device_scalar_pipeline_test()
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    const T          input[] = {T(3), T(0), T(0), T(4)};
    device_buffer<T> A(4), norm_result(1), trace_result(1), squared(1);
    ASSERT_EQ(A.upload(input), cudaSuccess);
    linalg::norm<T, CB>::apply(
        A.get(), 2, 2, 2, linalg::norm_type::FROBENIUS, nullptr, 0, norm_result.get(), stream);
    linalg::trace<T, CB>::apply(norm_result.get(), 1, 1, trace_result.get(), stream);
    linalg::multiplication<T, CB>::apply(false,
        false,
        1,
        1,
        1,
        trace_result.get(),
        1,
        trace_result.get(),
        1,
        squared.get(),
        1,
        nullptr,
        0,
        stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    T result{};
    ASSERT_EQ(squared.download(&result), cudaSuccess);
    EXPECT_NEAR(result, T(25), GpuTolerance<T>::value);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(MathGpu, DeviceScalarPipeline)
{
    device_scalar_pipeline_test<float>();
    device_scalar_pipeline_test<double>();
}

template <typename T> void device_spectral_outputs_test()
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    const T                    input[] = {T(4), T(0), T(0), T(2)};
    device_buffer<T>           A(4), condition(1), spectral_norm(1), pinv(4), det(1);
    device_buffer<linalg_long> rank_result(1);
    const auto                 rank_bytes = linalg::rank<T, CB>::workspace_size(2, 2);
    const auto norm_bytes = linalg::norm<T, CB>::workspace_size(2, 2, linalg::norm_type::TWO);
    const auto pinv_bytes = linalg::pseudo_inverse<T, CB>::workspace_size(2, 2);
    const auto mode       = linalg::linear_solver_type::LU_LINEAR_SOLVER;
    const auto det_bytes  = linalg::determinant<T, CB>::workspace_size(2, mode);
    device_buffer<unsigned char> workspace(
        std::max({rank_bytes, norm_bytes, pinv_bytes, det_bytes}));
    ASSERT_EQ(A.upload(input), cudaSuccess);
    linalg::rank<T, CB>::apply(
        A.get(), 2, 2, 2, T(-1), workspace.get(), rank_bytes, rank_result.get(), stream);
    linalg::condition_number<T, CB>::apply(
        A.get(), 2, 2, 2, workspace.get(), rank_bytes, condition.get(), stream);
    linalg::norm<T, CB>::apply(A.get(),
        2,
        2,
        2,
        linalg::norm_type::TWO,
        workspace.get(),
        norm_bytes,
        spectral_norm.get(),
        stream);
    linalg::pseudo_inverse<T, CB>::apply(
        2, 2, A.get(), 2, pinv.get(), 2, T(-1), workspace.get(), pinv_bytes, stream);
    linalg::determinant<T, CB>::apply(
        A.get(), nullptr, 2, mode, workspace.get(), det_bytes, det.get(), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    linalg_long rank{};
    T           cond{}, norm{}, determinant{};
    T           inverse[4]{};
    ASSERT_EQ(rank_result.download(&rank), cudaSuccess);
    ASSERT_EQ(condition.download(&cond), cudaSuccess);
    ASSERT_EQ(spectral_norm.download(&norm), cudaSuccess);
    ASSERT_EQ(pinv.download(inverse), cudaSuccess);
    ASSERT_EQ(det.download(&determinant), cudaSuccess);
    EXPECT_EQ(rank, static_cast<linalg_long>(2));
    EXPECT_NEAR(cond, T(2), GpuTolerance<T>::value);
    EXPECT_NEAR(norm, T(4), GpuTolerance<T>::value);
    EXPECT_NEAR(determinant, T(8), GpuTolerance<T>::value);
    EXPECT_NEAR(inverse[0], T(0.25), GpuTolerance<T>::value);
    EXPECT_NEAR(inverse[3], T(0.5), GpuTolerance<T>::value);
    EXPECT_NEAR(inverse[1], T(0), GpuTolerance<T>::value);
    EXPECT_NEAR(inverse[2], T(0), GpuTolerance<T>::value);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(MathGpu, DeviceSpectralOutputs)
{
    device_spectral_outputs_test<float>();
    device_spectral_outputs_test<double>();
}

TEST(MathGpu, DeviceFactorizationFailureStatus)
{
    const auto type  = linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR;
    const auto bytes = linalg::cholesky_decomposition<double, CB>::workspace_size(2, type);
    device_buffer<unsigned char> workspace(bytes);
    device_buffer<double>        A(4);
    device_buffer<int>           info(1);
    const double                 indefinite[] = {-1., 0., 0., 1.};
    ASSERT_EQ(A.upload(indefinite), cudaSuccess);
    linalg::cholesky_decomposition<double, CB>::apply(
        A.get(), 2, type, workspace.get(), bytes, info.get());
    int status{};
    ASSERT_EQ(info.download(&status), cudaSuccess);
    EXPECT_GT(status, 0);

    const auto                   lu_bytes = linalg::lu_decomposition<double, CB>::workspace_size(2);
    device_buffer<unsigned char> lu_workspace(lu_bytes);
    device_buffer<int>           pivot(2);
    const double                 singular[] = {0., 0., 0., 0.};
    ASSERT_EQ(A.upload(singular), cudaSuccess);
    linalg::lu_decomposition<double, CB>::apply(
        A.get(), 2, pivot.get(), lu_workspace.get(), lu_bytes, info.get());
    ASSERT_EQ(info.download(&status), cudaSuccess);
    EXPECT_GT(status, 0);
}

#endif  // LINALG_HAS_CUBLAS
