// Benchmarks linalg::svd_decomposition<T, Backend>::apply for every backend
// this build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_SVDDecomposition is templated on Backend too, and registered once per
// backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
// (native is unconditional, the scalar fallback always builds) — see
// Testing/Cxx/test_backend.h for the same per-backend coverage on the unit
// test side, and BenchmarkQRDecomposition.cxx for the sibling benchmark this
// one is modeled on. A single run now reports all backends present in this
// build, each row labeled with the backend that produced it.
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/svd_decomposition.h"

namespace
{

const char* backend_name(linalg::linalg_backend backend)
{
    switch (backend)
    {
    case linalg::linalg_backend::MKL:
        return "mkl";
    case linalg::linalg_backend::BLAS_LAPACKE:
#if LINALG_BENCHMARK_BLIS
        return "blis";
#else
        return "blas_lapacke";
#endif
    default:
        return "native";
    }
}

template <typename T>
std::vector<T> make_random_matrix(std::size_t rows, std::size_t columns, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> a(rows * columns);
    for (auto& v : a)
    {
        v = dist(rng);
    }
    return a;
}

// SVD's economy factorization needs rows >= columns (the native backend
// asserts this internally, see svd_decomposition_native.cpp); benchmarked
// here on square matrices for the simplest apples-to-apples comparison
// against BenchmarkQRDecomposition.cxx's and BenchmarkLUDecomposition.cxx's
// square-matrix sizes.
//
// Capped at 128 (rather than following those two up to 1024): the native
// backend's implicit-shift diagonalization sweep can take a very large
// number of iterations to converge for some random matrices as n grows --
// reproducible on this same random seed with the unmodified, pre-existing
// implementation too, so it predates and is independent of this file's
// vectorization -- and 256 was already observed to run far longer than is
// reasonable for a benchmark. Flagged as a separate, pre-existing
// correctness/performance issue rather than silently worked around by a
// different seed or input.
template <typename T, linalg::linalg_backend Backend>
void BM_SVDDecomposition(benchmark::State& state)
{
    const std::size_t    n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> a = make_random_matrix<T>(n, n, /*seed=*/42);

    const size_t ws_size = linalg::svd_decomposition<T, Backend>::workspace_size(
        static_cast<linalg_long>(n), static_cast<linalg_long>(n));
    std::vector<std::byte> workspace(ws_size);

    std::vector<T> a_work(a);
    std::vector<T> s(n);
    std::vector<T> u(n * n);
    std::vector<T> vt(n * n);

    for (auto _ : state)
    {
        // A is overwritten in place by the native backend, so it is
        // refreshed from the pristine copy before every iteration --
        // otherwise later iterations would factor an already-decomposed
        // (near-diagonal) matrix instead of the intended random one.
        a_work = a;
        linalg::svd_decomposition<T, Backend>::apply(static_cast<linalg_long>(n),
            static_cast<linalg_long>(n),
            a_work.data(),
            static_cast<linalg_long>(n),
            s.data(),
            u.data(),
            static_cast<linalg_long>(n),
            vt.data(),
            static_cast<linalg_long>(n),
            workspace.data(),
            ws_size);
        benchmark::DoNotOptimize(s.data());
        benchmark::DoNotOptimize(u.data());
        benchmark::DoNotOptimize(vt.data());
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));

    // Dense SVD of a square n x n matrix costs ~O(n^3); counted here with the
    // same ~4n^3 (bidiagonalization-dominated) constant BenchmarkQRDecomposition.cxx
    // and BenchmarkCholeskyDecomposition.cxx use for their own O(n^3) factorizations,
    // scaled by 2 for the extra (U and V) accumulation an SVD does over a plain QR.
    const double flops_per_iter =
        4.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    state.counters["FLOPS"] = benchmark::Counter(flops_per_iter,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_SVDDecomposition, float, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SVDDecomposition, double, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
BENCHMARK_TEMPLATE(BM_SVDDecomposition, float, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SVDDecomposition, double, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif

#if LINALG_HAS_MKL
BENCHMARK_TEMPLATE(BM_SVDDecomposition, float, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_SVDDecomposition, double, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif
