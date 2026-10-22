// GPU matrix-operation tests exercising the struct-template API under
// linalg_backend::METAL_BLAS. Companion to TestGpuMatrixOperations.cxx
// (CUDA_BLAS/ROCM_BLAS), but intentionally narrower: METAL_BLAS is float
// only (Apple GPUs have no native double type) and only covers
// multiplication, batched multiplication, transpose, norm, and trace, plus
// portable matrix logarithm and square root (see
// linalg_backend::METAL_BLAS's doc comment in include/common/backend.h) --
// cholesky/LU/QR/SVD/eigen/linear_solver/inversion/determinant/rank/
// pseudo_inverse/least_squares have no METAL_BLAS specialization, so they
// have no coverage here. Reuses the MathGpu test-suite name: like all
// backend-conditional tests in Testing/Cxx, this file contributes no TEST
// cases unless LINALG_HAS_METAL is set, and LINALG_ENABLE_METAL is mutually
// exclusive with LINALG_ENABLE_CUBLAS/LINALG_ENABLE_ROCBLAS (CMakeLists.txt),
// so at most one of the two MathGpu-suite files ever compiles any TEST case.
#if LINALG_HAS_METAL

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "dense_matrix_test_helper.h"
#include "include/matrix_operation/logarithm.h"
#include "include/matrix_operation/square_root.h"
#include "matrix_functions_test_cases.h"
#include "include/matrix_operation/multiplication.h"
#include "include/matrix_operation/multiplication_batched.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/trace.h"
#include "include/matrix_operation/transpose.h"
#include "gtest/gtest.h"
#include "test_reporting.h"

namespace
{
using linalg::linalg_backend;
using linalg_test::dense_matrix;

constexpr linalg_backend MB  = linalg_backend::METAL_BLAS;
constexpr float          tol = 5.e-3F;

// Minimal RAII page-aligned buffer: METAL_BLAS wraps every pointer it is
// given into an MTLBuffer via -newBufferWithBytesNoCopy:, which requires
// page alignment (see include/common/metal_handle.h); plain `new`/
// std::vector allocations are not guaranteed to be page-aligned, so tests
// need their own allocator here, the same way TestGpuMatrixOperations.cxx
// needs its own cudaMalloc/hipMalloc-backed device_buffer. Under Metal's
// unified memory this is also directly host-readable -- no separate
// upload/download step the way a discrete-GPU device_buffer needs.
template <typename T> class page_aligned_buffer
{
public:
    explicit page_aligned_buffer(std::size_t count) : count_(count)
    {
        if (count_ == 0)
        {
            return;
        }
        const std::size_t page  = static_cast<std::size_t>(getpagesize());
        const std::size_t bytes = count_ * sizeof(T);
        void*             raw   = nullptr;
        if (posix_memalign(&raw, page, bytes) != 0)
        {
            throw std::runtime_error("posix_memalign failed");
        }
        ptr_ = static_cast<T*>(raw);
        std::fill(ptr_, ptr_ + count_, T{});
    }
    ~page_aligned_buffer() { std::free(ptr_); }

    page_aligned_buffer(const page_aligned_buffer&)            = delete;
    page_aligned_buffer& operator=(const page_aligned_buffer&) = delete;

    T*       get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    std::size_t count_;
    T*          ptr_ = nullptr;
};

dense_matrix<float> random_matrix(
    std::size_t rows, std::size_t cols, std::default_random_engine& gen)
{
    std::uniform_real_distribution<float> distribution(-5.F, 5.F);
    dense_matrix<float>                   M(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            M(i, j) = distribution(gen);
        }
    }
    return M;
}

float max_abs_diff(const dense_matrix<float>& a, const dense_matrix<float>& b)
{
    float max_error = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < a.cols(); ++j)
        {
            max_error = std::fmax(max_error, std::fabs(a(i, j) - b(i, j)));
        }
    }
    return max_error;
}

void matrix_multiplication_gpu_test(
    int rows, int columns, int depth, bool transpose_a, bool transpose_b)
{
    auto a_rows = transpose_a ? depth : rows;
    auto a_cols = transpose_a ? rows : depth;
    auto b_rows = transpose_b ? columns : depth;
    auto b_cols = transpose_b ? depth : columns;

    std::default_random_engine generator;
    auto A = random_matrix((std::size_t)a_rows, (std::size_t)a_cols, generator);
    auto B = random_matrix((std::size_t)b_rows, (std::size_t)b_cols, generator);

    page_aligned_buffer<float> dA((std::size_t)(a_rows * a_cols));
    page_aligned_buffer<float> dB((std::size_t)(b_rows * b_cols));
    page_aligned_buffer<float> dC((std::size_t)(rows * columns));
    std::copy(A.data(), A.data() + A.rows() * A.cols(), dA.get());
    std::copy(B.data(), B.data() + B.rows() * B.cols(), dB.get());

    linalg::multiplication<float, MB>::apply(transpose_a,
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

    dense_matrix<float> C((std::size_t)rows, (std::size_t)columns);
    std::copy(dC.get(), dC.get() + (std::size_t)(rows * columns), C.data());

    dense_matrix<float> expected =
        matmul(transpose_a ? transpose(A) : A, transpose_b ? transpose(B) : B);

    EXPECT_LE(max_abs_diff(C, expected), tol);
}

}  // namespace

TEST(MathGpu, MatrixMultiplication)
{
    const int rows = 37, columns = 53, depth = 29;

    matrix_multiplication_gpu_test(rows, columns, depth, false, false);
    matrix_multiplication_gpu_test(rows, columns, depth, true, false);
    matrix_multiplication_gpu_test(rows, columns, depth, false, true);
    matrix_multiplication_gpu_test(rows, columns, depth, true, true);
}

TEST(MathGpu, MatrixTranspose)
{
    const std::size_t rows = 41, columns = 23;

    std::default_random_engine generator;
    auto                       A = random_matrix(rows, columns, generator);

    page_aligned_buffer<float> dA(rows * columns);
    std::copy(A.data(), A.data() + A.rows() * A.cols(), dA.get());

    const auto ws_size =
        linalg::transpose<float, MB>::workspace_size((linalg_long)rows, (linalg_long)columns);
    page_aligned_buffer<unsigned char> dws(ws_size);
    linalg::transpose<float, MB>::apply(
        (linalg_long)rows, (linalg_long)columns, dA.get(), dws.get(), ws_size);

    dense_matrix<float> T(columns, rows);
    std::copy(dA.get(), dA.get() + rows * columns, T.data());
    EXPECT_LE(max_abs_diff(T, transpose(A)), tol);
}

TEST(MathGpu, BatchedMatrixMultiplication)
{
    const int dim = 9, count = 6;

    std::default_random_engine            generator;
    std::vector<float>                    a((std::size_t)(dim * dim * count));
    std::vector<float>                    b((std::size_t)(dim * dim * count));
    std::uniform_real_distribution<float> distribution(-5.F, 5.F);
    for (auto& v : a)
        v = distribution(generator);
    for (auto& v : b)
        v = distribution(generator);

    page_aligned_buffer<float> da(a.size());
    page_aligned_buffer<float> db(b.size());
    page_aligned_buffer<float> dc(a.size());
    std::copy(a.begin(), a.end(), da.get());
    std::copy(b.begin(), b.end(), db.get());

    linalg::batched_matrix_multiplication(dim, count, da.get(), db.get(), dc.get());

    float max_error = 0;
    for (int mat = 0; mat < count; ++mat)
    {
        const float* Ai = a.data() + (std::size_t)(mat * dim * dim);
        const float* Bi = b.data() + (std::size_t)(mat * dim * dim);
        const float* Ci = dc.get() + (std::size_t)(mat * dim * dim);
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                float sum = 0.F;
                for (int k = 0; k < dim; ++k)
                {
                    sum += Ai[i * dim + k] * Bi[k * dim + j];
                }
                max_error = std::fmax(max_error, std::fabs(sum - Ci[i * dim + j]));
            }
        }
    }
    EXPECT_LE(max_error, tol * dim);
}

TEST(MathGpu, MatrixNorm)
{
    const std::size_t rows = 6, columns = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix(rows, columns, generator);

    page_aligned_buffer<float> dA(rows * columns);
    std::copy(A.data(), A.data() + A.rows() * A.cols(), dA.get());

    float frob_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            frob_expected += A(i, j) * A(i, j);
    frob_expected = std::sqrt(frob_expected);

    float one_expected = 0;
    for (std::size_t j = 0; j < columns; ++j)
    {
        float s = 0;
        for (std::size_t i = 0; i < rows; ++i)
            s += std::fabs(A(i, j));
        one_expected = std::max(one_expected, s);
    }

    float inf_expected = 0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        float s = 0;
        for (std::size_t j = 0; j < columns; ++j)
            s += std::fabs(A(i, j));
        inf_expected = std::max(inf_expected, s);
    }

    const auto frob_ws_size = linalg::norm<float, MB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::FROBENIUS);
    page_aligned_buffer<unsigned char> frob_dws(frob_ws_size);
    const auto one_ws_size = linalg::norm<float, MB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::ONE);
    page_aligned_buffer<unsigned char> one_dws(one_ws_size);
    const auto inf_ws_size = linalg::norm<float, MB>::workspace_size(
        (linalg_long)rows, (linalg_long)columns, linalg::norm_type::INFINITY_NORM);
    page_aligned_buffer<unsigned char> inf_dws(inf_ws_size);

    const auto frob = linalg::norm<float, MB>::apply(dA.get(),
        (linalg_long)rows, (linalg_long)columns, (linalg_long)columns,
        linalg::norm_type::FROBENIUS, frob_dws.get(), frob_ws_size);
    const auto one = linalg::norm<float, MB>::apply(dA.get(),
        (linalg_long)rows, (linalg_long)columns, (linalg_long)columns,
        linalg::norm_type::ONE, one_dws.get(), one_ws_size);
    const auto inf_norm = linalg::norm<float, MB>::apply(dA.get(),
        (linalg_long)rows, (linalg_long)columns, (linalg_long)columns,
        linalg::norm_type::INFINITY_NORM, inf_dws.get(), inf_ws_size);

    EXPECT_NEAR(frob, frob_expected, tol);
    EXPECT_NEAR(one, one_expected, tol);
    EXPECT_NEAR(inf_norm, inf_expected, tol);
}

TEST(MathGpu, MatrixTrace)
{
    const std::size_t n = 5;
    std::default_random_engine generator;
    auto                       A = random_matrix(n, n, generator);

    float expected = 0;
    for (std::size_t i = 0; i < n; ++i)
        expected += A(i, i);

    page_aligned_buffer<float> dA(n * n);
    std::copy(A.data(), A.data() + A.rows() * A.cols(), dA.get());

    const auto tr = linalg::trace<float, MB>::apply(dA.get(), (linalg_int)n, (linalg_int)n);
    EXPECT_NEAR(tr, expected, tol);
}

// Every covered struct-API entry point throws on a null required pointer or
// a non-positive dimension -- mirrors MathGpu.InputValidation in
// TestGpuMatrixOperations.cxx, scoped to METAL_BLAS's narrower op coverage.
TEST(MathGpu, InputValidation)
{
    const int n = 3;
    page_aligned_buffer<float> dA((std::size_t)(n * n));

    EXPECT_ANY_THROW((linalg::multiplication<float, MB>::apply(false, false, n, n, n,
        static_cast<const float*>(nullptr), n, dA.get(), n, dA.get(), n, nullptr, 0)));
    EXPECT_ANY_THROW((linalg::multiplication<float, MB>::apply(
        false, false, 0, n, n, dA.get(), n, dA.get(), n, dA.get(), n, nullptr, 0)));

    EXPECT_ANY_THROW((linalg::transpose<float, MB>::apply(
        (linalg_long)n, (linalg_long)n, static_cast<float*>(nullptr), nullptr, 0)));
    EXPECT_ANY_THROW(
        (linalg::transpose<float, MB>::apply(0LL, (linalg_long)n, dA.get(), nullptr, 0)));

    EXPECT_ANY_THROW((linalg::trace<float, MB>::apply(static_cast<const float*>(nullptr), n, n)));
    EXPECT_ANY_THROW((linalg::trace<float, MB>::apply(dA.get(), 0, n)));

    EXPECT_ANY_THROW((linalg::norm<float, MB>::apply(static_cast<const float*>(nullptr),
        (linalg_long)n, (linalg_long)n, (linalg_long)n, linalg::norm_type::FROBENIUS, nullptr, 0)));
    EXPECT_ANY_THROW((linalg::norm<float, MB>::apply(dA.get(),
        0LL, (linalg_long)n, (linalg_long)n, linalg::norm_type::FROBENIUS, nullptr, 0)));

    EXPECT_ANY_THROW((linalg::batched_matrix_multiplication(
        n, 2, static_cast<const float*>(nullptr), dA.get(), dA.get())));
    EXPECT_ANY_THROW(
        (linalg::batched_matrix_multiplication(0, 2, dA.get(), dA.get(), dA.get())));
}

TEST(MathGpu, MatrixFunctions)
{
    linalg_test::for_each_matrix_function_case(
        [](int n, const double* input, const double* expected, bool is_log)
        {
            SCOPED_TRACE(is_log ? "logarithm" : "square_root");
            const int  stride = n + 2;
            const auto bytes  = linalg::logarithm<float, MB>::workspace_size(n);
            page_aligned_buffer<unsigned char> workspace(bytes);
            page_aligned_buffer<float>         a(n * stride), result(n * stride);
            std::vector<float>                 host(n * stride, -123.F);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    host[i * stride + j] = static_cast<float>(input[i * n + j]);
                }
            }
            for (bool in_place : {false, true})
            {
                std::copy(host.begin(), host.end(), a.get());
                std::copy(host.begin(), host.end(), result.get());
                float* destination = in_place ? a.get() : result.get();
                if (is_log)
                {
                    linalg::logarithm<float, MB>::apply(
                        a.get(), n, stride, destination, stride, workspace.get(), bytes);
                }
                else
                {
                    linalg::square_root<float, MB>::apply(
                        a.get(), n, stride, destination, stride, workspace.get(), bytes);
                }
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        EXPECT_NEAR(destination[i * stride + j],
                            expected[i * n + j],
                            8.e-4 * std::max(1., std::fabs(expected[i * n + j])));
                    }
                    EXPECT_EQ(destination[i * stride + n], -123.F);
                    EXPECT_EQ(destination[i * stride + n + 1], -123.F);
                }
                if (!in_place)
                {
                    EXPECT_TRUE(std::equal(host.begin(), host.end(), a.get()));
                }
            }
        });
    const auto                         bytes = linalg::logarithm<float, MB>::workspace_size(2);
    page_aligned_buffer<unsigned char> workspace(bytes);
    page_aligned_buffer<float>         a(4), result(4);
    a.get()[0] = -1.F;
    a.get()[3] = 1.F;
    linalg::logarithm<float, MB>::apply(a.get(), 2, 2, result.get(), 2, workspace.get(), bytes);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(std::isnan(result.get()[i]));
    }
    linalg::square_root<float, MB>::apply(a.get(), 2, 2, result.get(), 2, workspace.get(), bytes);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(std::isnan(result.get()[i]));
    }
}

#endif  // LINALG_HAS_METAL
