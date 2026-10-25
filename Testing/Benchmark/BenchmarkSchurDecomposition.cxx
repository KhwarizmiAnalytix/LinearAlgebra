// Full complex Schur form and unitary vectors, including split-plane output.
// Each backend receives the same real, dense nonsymmetric matrix.
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <Eigen/Eigenvalues>

#include "include/common/backend.h"
#include "include/matrix_operation/schur_decomposition.h"

namespace
{
template <typename T>
using RealMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using ComplexMatrix =
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T> std::vector<T> make_matrix(linalg_int n)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<T> distribution(T(-1), T(1));
    std::vector<T> input(static_cast<std::size_t>(n) * n);
    for (auto& value : input)
        value = distribution(rng);
    return input;
}

// Check the actual benchmark input before timing: A = Q S Q*, Q*Q = I,
// and S upper triangular. This also rejects nonfinite outputs.
template <typename T>
bool valid_schur(const std::vector<T>& input, linalg_int n,
    const RealMatrix<T>& sr, const RealMatrix<T>& si,
    const RealMatrix<T>& qr, const RealMatrix<T>& qi)
{
    const ComplexMatrix<T> s = sr.template cast<std::complex<T>>() +
                              std::complex<T>(0, 1) * si.template cast<std::complex<T>>();
    const ComplexMatrix<T> q = qr.template cast<std::complex<T>>() +
                              std::complex<T>(0, 1) * qi.template cast<std::complex<T>>();
    const ComplexMatrix<T> a = Eigen::Map<const RealMatrix<T>>(input.data(), n, n)
                                  .template cast<std::complex<T>>();
    const T tolerance = T(256) * T(n) * std::numeric_limits<T>::epsilon();
    const T residual = (a - q * s * q.adjoint()).norm() / a.norm();
    const T orthogonality =
        (q.adjoint() * q - ComplexMatrix<T>::Identity(n, n)).norm() / std::sqrt(T(n));
    const T lower = s.template triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm();
    return s.allFinite() && q.allFinite() && std::isfinite(residual) &&
           residual <= tolerance && orthogonality <= tolerance && lower <= tolerance * s.norm();
}

template <typename T, linalg::linalg_backend Backend>
void BM_SchurDecomposition(benchmark::State& state)
{
    const auto n = static_cast<linalg_int>(state.range(0));
    const auto input = make_matrix<T>(n);
    RealMatrix<T> sr(n, n), si(n, n), qr(n, n), qi(n, n);
    const auto bytes = linalg::schur_decomposition<T, Backend>::workspace_size(n);
    std::vector<std::byte> workspace(bytes);
    const auto compute = [&] {
        linalg::schur_decomposition<T, Backend>::apply(input.data(), n, n,
            sr.data(), si.data(), n, qr.data(), qi.data(), n, workspace.data(), bytes);
    };
    try
    {
        compute();
        if (!valid_schur(input, n, sr, si, qr, qi))
        {
            state.SkipWithError("Schur reconstruction or unitarity check failed");
            return;
        }
        for (auto _ : state)
        {
            compute();
            benchmark::ClobberMemory();
        }
    }
    catch (const std::exception& error)
    {
        state.SkipWithError(error.what());
        return;
    }
    // CPU Schur backends currently share the portable implementation.
    state.SetLabel("portable complex Schur with unitary vectors");
    state.SetComplexityN(n);
}

template <typename T> void BM_EigenSchurDecomposition(benchmark::State& state)
{
    const auto n = static_cast<linalg_int>(state.range(0));
    const auto input = make_matrix<T>(n);
    const Eigen::Map<const RealMatrix<T>> original(input.data(), n, n);
    ComplexMatrix<T> work(n, n);
    Eigen::ComplexSchur<ComplexMatrix<T>> solver(n);
    RealMatrix<T> sr(n, n), si(n, n), qr(n, n), qi(n, n);
    const auto compute = [&] {
        work = original.template cast<std::complex<T>>();
        solver.compute(work, true);
        sr = solver.matrixT().real();
        si = solver.matrixT().imag();
        qr = solver.matrixU().real();
        qi = solver.matrixU().imag();
    };
    compute();
    if (solver.info() != Eigen::Success || !valid_schur(input, n, sr, si, qr, qi))
    {
        state.SkipWithError("Eigen complex Schur validation failed");
        return;
    }
    for (auto _ : state)
    {
        compute();
        if (solver.info() != Eigen::Success)
        {
            state.SkipWithError("Eigen complex Schur failed to converge");
            break;
        }
        benchmark::DoNotOptimize(sr.data());
        benchmark::DoNotOptimize(si.data());
        benchmark::DoNotOptimize(qr.data());
        benchmark::DoNotOptimize(qi.data());
        benchmark::ClobberMemory();
    }
    state.SetLabel("eigen complex Schur with unitary vectors and split planes");
    state.SetComplexityN(n);
}

void sizes(benchmark::Benchmark* registration)
{
    registration->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(96)->Arg(128)
        ->Unit(benchmark::kMicrosecond)->MinTime(0.1)->Repetitions(3)
        ->ReportAggregatesOnly(true)->Threads(1);
}
}  // namespace

BENCHMARK_TEMPLATE(BM_SchurDecomposition, float, linalg::linalg_backend::NATIVE)->Apply(sizes);
BENCHMARK_TEMPLATE(BM_SchurDecomposition, double, linalg::linalg_backend::NATIVE)->Apply(sizes);
#if LINALG_HAS_BLAS
BENCHMARK_TEMPLATE(BM_SchurDecomposition, float, linalg::linalg_backend::BLAS_LAPACKE)->Apply(sizes);
BENCHMARK_TEMPLATE(BM_SchurDecomposition, double, linalg::linalg_backend::BLAS_LAPACKE)->Apply(sizes);
#endif
#if LINALG_HAS_MKL
BENCHMARK_TEMPLATE(BM_SchurDecomposition, float, linalg::linalg_backend::MKL)->Apply(sizes);
BENCHMARK_TEMPLATE(BM_SchurDecomposition, double, linalg::linalg_backend::MKL)->Apply(sizes);
#endif
BENCHMARK_TEMPLATE(BM_EigenSchurDecomposition, float)->Apply(sizes);
BENCHMARK_TEMPLATE(BM_EigenSchurDecomposition, double)->Apply(sizes);
