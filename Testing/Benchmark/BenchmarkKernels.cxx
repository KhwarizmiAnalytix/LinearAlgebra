// Additional comparisons use identical deterministic row-major inputs and
// preallocated outputs/workspaces. Input refresh and upfront LU are untimed.
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "include/matrix_operation/inversion.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "include/matrix_operation/norm.h"
#include "include/matrix_operation/transpose.h"
#include "include/matrix_operation/triangular_solver.h"

namespace
{
using linalg::linalg_backend;
using linalg::linear_solver_type;
using linalg::norm_type;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T> Matrix<T> make_input(linalg_int rows, linalg_int columns)
{
    // NOLINTNEXTLINE(bugprone-random-generator-seed): comparisons require reproducible inputs.
    std::mt19937                      rng(42);
    std::uniform_real_distribution<T> distribution(T(-1), T(1));
    Matrix<T>                         result(rows, columns);
    for (Eigen::Index i = 0; i < result.size(); ++i)
    {
        result.data()[i] = distribution(rng);
    }
    // Strict diagonal dominance keeps LU inputs nonsingular even when native
    // pivoting is disabled; Eigen and LAPACK still use partial pivoting.
    if (rows == columns)
    {
        result.diagonal().array() += T(2 * rows);
    }
    return result;
}

template <typename T, linalg_backend Backend, bool Reference = false>
void transpose_case(benchmark::State& state)
{
    const auto                 rows = state.range(0), columns = state.range(1);
    const Matrix<T>            input = make_input<T>(rows, columns);
    Matrix<T>                  work  = input;
    std::vector<unsigned char> workspace(
        linalg::transpose<T, Backend>::workspace_size(rows, columns));
    for (auto _ : state)
    {
        state.PauseTiming();
        work = input;
        state.ResumeTiming();
        if constexpr (Reference)
        {
            work.transposeInPlace();
        }
        else
        {
            linalg::transpose<T, Backend>::apply(
                rows, columns, work.data(), workspace.data(), workspace.size());
        }
        benchmark::DoNotOptimize(work.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * rows * columns * 2 * sizeof(T));
}

template <typename T, linalg_backend Backend, norm_type Type, bool Reference = false>
void norm_case(benchmark::State& state)
{
    const auto      n     = state.range(0);
    const Matrix<T> input = make_input<T>(n, n);
    for (auto _ : state)
    {
        T result;
        if constexpr (!Reference)
        {
            result = linalg::norm<T, Backend>::apply(input.data(), n, n, n, Type, nullptr, 0);
        }
        else if constexpr (Type == norm_type::FROBENIUS)
        {
            result = input.norm();
        }
        else if constexpr (Type == norm_type::ONE)
        {
            result = input.cwiseAbs().colwise().sum().maxCoeff();
        }
        else
        {
            result = input.cwiseAbs().rowwise().sum().maxCoeff();
        }
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

template <typename T, linalg_backend Backend, bool Reuse, bool Invert, bool Reference = false>
void solve_case(benchmark::State& state)
{
    const auto      n     = state.range(0);
    const Matrix<T> input = make_input<T>(n, n);
    Matrix<T>       work(n, n), inverse(n, n);
    if constexpr (!Reference || !Reuse)
    {
        work = input;
    }
    const Vector<T>         rhs      = Vector<T>::Ones(n);
    Vector<T>               solution = rhs;
    std::vector<linalg_int> pivot(n);
    const auto              mode =
        Reuse ? linear_solver_type::LU_UPFRONT_LINEAR_SOLVER : linear_solver_type::LU_LINEAR_SOLVER;
    const auto bytes = Invert ? linalg::inversion<T, Backend>::workspace_size(n, mode)
                              : linalg::linear_solver<T, Backend>::workspace_size(n, mode);
    std::vector<unsigned char> workspace(bytes);
    std::vector<unsigned char> factor_workspace(
        linalg::lu_decomposition<T, Backend>::workspace_size(n));
    Eigen::PartialPivLU<Matrix<T>> solver(n);
    if constexpr (Reuse)
    {
        if constexpr (Reference)
        {
            solver.compute(input);
        }
        else if (!linalg::lu_decomposition<T, Backend>::apply(work.data(),
                     n,
                     pivot.data(),
                     factor_workspace.data(),
                     factor_workspace.size()))
        {
            state.SkipWithError("Upfront LU failed");
            return;
        }
    }
    for (auto _ : state)
    {
        state.PauseTiming();
        if constexpr (!Reuse)
        {
            work = input;
        }
        solution = rhs;
        state.ResumeTiming();
        if constexpr (Reference)
        {
            if constexpr (!Reuse)
            {
                solver.compute(work);
            }
            if constexpr (Invert)
            {
                inverse = solver.inverse();
                benchmark::DoNotOptimize(inverse.data());
            }
            else
            {
                solution = solver.solve(rhs);
                benchmark::DoNotOptimize(solution.data());
            }
        }
        else if constexpr (Invert)
        {
            linalg::inversion<T, Backend>::apply(
                work.data(), pivot.data(), n, mode, workspace.data(), bytes);
            benchmark::DoNotOptimize(work.data());
        }
        else
        {
            linalg::linear_solver<T, Backend>::apply(
                work.data(), pivot.data(), n, solution.data(), mode, workspace.data(), bytes);
            benchmark::DoNotOptimize(solution.data());
        }
        benchmark::ClobberMemory();
    }
}

template <typename T, linalg_backend Backend, bool Upper, bool Transpose, bool Reference = false>
void triangular_case(benchmark::State& state)
{
    const auto                 n = state.range(0), nrhs = state.range(1);
    const Matrix<T>            input = make_input<T>(n, n);
    const Matrix<T>            rhs   = make_input<T>(n, nrhs);
    Matrix<T>                  output(n, nrhs);
    std::vector<unsigned char> workspace(
        linalg::triangular_solver<T, Backend>::workspace_size(n, nrhs));
    for (auto _ : state)
    {
        if constexpr (Reference)
        {
            output = rhs;
            if constexpr (Transpose)
            {
                input.transpose()
                    .template triangularView<Upper ? Eigen::Lower : Eigen::Upper>()
                    .solveInPlace(output);
            }
            else
            {
                input.template triangularView<Upper ? Eigen::Upper : Eigen::Lower>().solveInPlace(
                    output);
            }
        }
        else
        {
            linalg::triangular_solver<T, Backend>::apply(Upper,
                Transpose,
                false,
                n,
                nrhs,
                input.data(),
                n,
                rhs.data(),
                nrhs,
                output.data(),
                nrhs,
                workspace.data(),
                workspace.size());
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
}

template <typename Benchmark>
void configure(Benchmark* item, bool rectangular = false, bool rhs = false)
{
    if (rectangular)
    {
        item->Args({32, 32})
            ->Args({256, 256})
            ->Args({1024, 1024})
            ->Args({64, 256})
            ->Args({256, 64});
    }
    else if (rhs)
    {
        item->Args({32, 1})->Args({256, 1})->Args({256, 16})->Args({1024, 16});
    }
    else
    {
        item->Arg(32)->Arg(256)->Arg(1024);
    }
    item->Unit(benchmark::kMicrosecond)
        ->MinTime(0.1)
        ->Repetitions(3)
        ->ReportAggregatesOnly(true)
        ->Threads(1);
}

template <typename T, linalg_backend Backend, bool Reference = false>
void register_cases(const std::string& scalar, const std::string& backend)
{
    const std::string prefix = Reference ? "BM_Eigen" : "BM_";
    const std::string suffix =
        "<" + scalar + (Reference ? "" : ", linalg::linalg_backend::" + backend) + ">";
    const auto add =
        [&](const std::string& kernel, auto function, bool rectangular = false, bool rhs = false)
    {
        configure(benchmark::RegisterBenchmark((prefix + kernel + suffix).c_str(), function),
            rectangular,
            rhs);
    };
    add("Transpose", &transpose_case<T, Backend, Reference>, true);
    add("NormFrobenius", &norm_case<T, Backend, norm_type::FROBENIUS, Reference>);
    add("NormOne", &norm_case<T, Backend, norm_type::ONE, Reference>);
    add("NormInfinity", &norm_case<T, Backend, norm_type::INFINITY_NORM, Reference>);
    add("LinearSolve", &solve_case<T, Backend, false, false, Reference>);
    add("FactorizedSolve", &solve_case<T, Backend, true, false, Reference>);
    add("Inversion", &solve_case<T, Backend, false, true, Reference>);
    add("TriangularLower", &triangular_case<T, Backend, false, false, Reference>, false, true);
    add("TriangularUpper", &triangular_case<T, Backend, true, false, Reference>, false, true);
    add("TriangularLowerTranspose",
        &triangular_case<T, Backend, false, true, Reference>,
        false,
        true);
    add("TriangularUpperTranspose",
        &triangular_case<T, Backend, true, true, Reference>,
        false,
        true);
}

// Google Benchmark registers cases before main; registration failures are fatal.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
const bool registered = []
{
    register_cases<float, linalg_backend::NATIVE>("float", "NATIVE");
    register_cases<double, linalg_backend::NATIVE>("double", "NATIVE");
    register_cases<float, linalg_backend::NATIVE, true>("float", "");
    register_cases<double, linalg_backend::NATIVE, true>("double", "");
#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    register_cases<float, linalg_backend::BLAS_LAPACKE>("float", "BLAS_LAPACKE");
    register_cases<double, linalg_backend::BLAS_LAPACKE>("double", "BLAS_LAPACKE");
#endif
#if LINALG_HAS_MKL
    register_cases<float, linalg_backend::MKL>("float", "MKL");
    register_cases<double, linalg_backend::MKL>("double", "MKL");
#endif
    return true;
}();
}  // namespace
