// Benchmarks linalg::cholesky_decomposition<T, Backend>::apply for every
// backend this build actually compiled in.
//
// The library never defaults to a backend — every linalg::*<T, Backend> call
// requires Backend to be named explicitly. This benchmark mirrors that:
// BM_CholeskyDecomposition is templated on Backend too, and registered once
// per backend gated on LINALG_HAS_MKL / LINALG_HAS_BLAS / LINALG_HAS_LAPACKE
// (native is unconditional, the scalar fallback always builds) — see
// Testing/Cxx/test_backend.h for the same per-backend coverage on the unit
// test side. A single run now reports all backends present in this build,
// each row labeled with the backend that produced it.
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/cholesky_decomposition.h"

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

// Random row-major n x n SPD matrix: A = M * M^T + n * I, SPD for any real M.
template <typename T> std::vector<T> make_spd_matrix(std::size_t n, unsigned seed)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1), static_cast<T>(1));

    std::vector<T> m(n * n);
    for (auto& v : m)
    {
        v = dist(rng);
    }

    std::vector<T> a(n * n, T{});
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            T sum = T{};
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += m[i * n + k] * m[j * n + k];
            }
            a[i * n + j] = sum;
        }
        a[i * n + i] += static_cast<T>(n);
    }
    return a;
}

template <typename T, linalg::linalg_backend Backend>
void BM_CholeskyDecomposition(benchmark::State& state)
{
    const std::size_t    n   = static_cast<std::size_t>(state.range(0));
    const std::vector<T> spd = make_spd_matrix<T>(n, /*seed=*/42);
    std::vector<T>       work(n * n);

    for (auto _ : state)
    {
        state.PauseTiming();
        work = spd;
        state.ResumeTiming();

        bool ok = linalg::cholesky_decomposition<T, Backend>::apply(work.data(),
            static_cast<linalg_int>(n),
            linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
        benchmark::DoNotOptimize(ok);
    }

    state.SetLabel(backend_name(Backend));
    state.SetComplexityN(static_cast<std::int64_t>(n));

    // Cholesky factorization of an n x n SPD matrix costs ~n^3/3 flops.
    const double flops_per_iter =
        static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n) / 3.0;
    state.counters["FLOPS"] = benchmark::Counter(flops_per_iter,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, float, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, double, linalg::linalg_backend::NATIVE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, float, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, double, linalg::linalg_backend::BLAS_LAPACKE)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif

#if LINALG_HAS_MKL
BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, float, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, double, linalg::linalg_backend::MKL)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true)
    ->Threads(1);
#endif
