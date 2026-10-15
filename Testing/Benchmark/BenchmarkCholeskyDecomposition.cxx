// Benchmarks linalg::cholesky_decomposition<T, linalg::kActiveBackend>::apply.
//
// kActiveBackend is a compile-time choice (include/common/backend.h): a
// single build only ever contains native, blas_lapacke, or mkl, never all
// three. To compare them, configure/build this target once per backend
// (LINALG_ENABLE_MKL=OFF/LINALG_ENABLE_BLAS=OFF for native, then
// LINALG_ENABLE_BLAS=ON, then LINALG_ENABLE_MKL=ON) and compare the three
// runs' output — each run labels its results with the backend that produced
// them.
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

const char* backend_name()
{
    switch (linalg::kActiveBackend)
    {
    case linalg::linalg_backend::mkl:
        return "mkl";
    case linalg::linalg_backend::blas_lapacke:
        return "blas_lapacke";
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

template <typename T> void BM_CholeskyDecomposition(benchmark::State& state)
{
    const std::size_t   n   = static_cast<std::size_t>(state.range(0));
    const std::vector<T> spd = make_spd_matrix<T>(n, /*seed=*/42);
    std::vector<T>       work(n * n);

    for (auto _ : state)
    {
        state.PauseTiming();
        work = spd;
        state.ResumeTiming();

        bool ok = linalg::cholesky_decomposition<T, linalg::kActiveBackend>::apply(work.data(),
            static_cast<linalg_int>(n), linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR);
        benchmark::DoNotOptimize(ok);
    }

    state.SetLabel(backend_name());
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, float)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_CholeskyDecomposition, double)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
