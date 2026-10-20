// Profile symmetric_eigen_backend and general_eigen_backend across every
// available CPU backend (native, blas_lapacke, mkl) using
// KhwarizmiAnalytix/Profiler instrumentation.
//
// Run:
//   ./bin/profile_linalg_eigen [n]
//
// where n is the matrix order (default 128).  The program also sweeps n/2
// and 2*n to give three data points per backend.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "profiler.h"

#include "include/common/backend.h"
#include "include/common/types.h"
#include "include/matrix_operation/eigen.h"

namespace
{

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

template <typename T>
std::vector<T> make_random_matrix(std::size_t rows, std::size_t cols, unsigned seed = 42)
{
    std::mt19937                      rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    std::vector<T>                    m(rows * cols);
    for (auto& v : m)
    {
        v = dist(rng);
    }
    return m;
}

// Returns a symmetric positive-definite n x n matrix: B * B^T + n * I.
template <typename T>
std::vector<T> make_spd_matrix(linalg_int n, unsigned seed = 42)
{
    auto           B = make_random_matrix<T>(static_cast<std::size_t>(n), static_cast<std::size_t>(n), seed);
    std::vector<T> A(static_cast<std::size_t>(n * n), T(0));
    for (linalg_int i = 0; i < n; ++i)
    {
        for (linalg_int j = 0; j < n; ++j)
        {
            T sum = T(0);
            for (linalg_int k = 0; k < n; ++k)
            {
                sum += B[static_cast<std::size_t>(i * n + k)] *
                       B[static_cast<std::size_t>(j * n + k)];
            }
            A[static_cast<std::size_t>(i * n + j)] = sum;
        }
        A[static_cast<std::size_t>(i * n + i)] += static_cast<T>(n);
    }
    return A;
}

// ---------------------------------------------------------------------------
// Per-size eigen profiles
// ---------------------------------------------------------------------------

template <linalg::linalg_backend Backend>
void profile_eigen_at_size(linalg_int n)
{
    // symmetric_eigen: eigenvalues + eigenvectors of SPD matrix
    {
        PROFILER_SCOPE("symmetric/with_vectors");
        auto                A            = make_spd_matrix<double>(n);
        std::vector<double> eigenvalues(static_cast<std::size_t>(n));
        std::vector<double> eigenvectors(static_cast<std::size_t>(n * n));
        const auto          ws           = linalg::symmetric_eigen_backend<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        linalg::symmetric_eigen_backend<double, Backend>::apply(
            A.data(), n, n, eigenvalues.data(), eigenvectors.data(), n, workspace.data(), ws);
    }

    // symmetric_eigen: eigenvalues only (eigenvectors buffer still provided but
    // we time the same call; the backend decides what to skip internally)
    {
        PROFILER_SCOPE("symmetric/eigenvalues_only");
        auto                A          = make_spd_matrix<double>(n, 43);
        std::vector<double> eigenvalues(static_cast<std::size_t>(n));
        const auto          ws         = linalg::symmetric_eigen_backend<double, Backend>::workspace_size(n);
        std::vector<unsigned char> workspace(ws);
        // Pass nullptr for eigenvectors — backend must accept a null pointer
        // when only eigenvalues are needed (matches the general_eigen_backend
        // convention; symmetric_eigen_backend's apply signature requires a
        // non-null eigenvectors pointer, so reuse a small buffer instead).
        std::vector<double> dummy_vecs(static_cast<std::size_t>(n * n));
        linalg::symmetric_eigen_backend<double, Backend>::apply(
            A.data(), n, n, eigenvalues.data(), dummy_vecs.data(), n, workspace.data(), ws);
    }

    // general_eigen: eigenvalues only (no eigenvectors)
    {
        PROFILER_SCOPE("general/eigenvalues_only");
        auto                A         = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        std::vector<double> ev_real(static_cast<std::size_t>(n));
        std::vector<double> ev_imag(static_cast<std::size_t>(n));
        const auto          ws        = linalg::general_eigen_backend<double, Backend>::workspace_size(n, false);
        std::vector<unsigned char> workspace(ws);
        linalg::general_eigen_backend<double, Backend>::apply(
            A.data(), n, n, ev_real.data(), ev_imag.data(), nullptr, n, workspace.data(), ws);
    }

    // general_eigen: eigenvalues + right eigenvectors
    // The native scalar backend raises an exception when a random matrix has
    // complex eigenvalues; catch it and note the skip so the rest of the
    // profile continues unaffected.
    try
    {
        PROFILER_SCOPE("general/with_vectors");
        auto                A            = make_random_matrix<double>(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 43);
        std::vector<double> ev_real(static_cast<std::size_t>(n));
        std::vector<double> ev_imag(static_cast<std::size_t>(n));
        std::vector<double> eigenvectors(static_cast<std::size_t>(n * n));
        const auto          ws           = linalg::general_eigen_backend<double, Backend>::workspace_size(n, true);
        std::vector<unsigned char> workspace(ws);
        linalg::general_eigen_backend<double, Backend>::apply(
            A.data(), n, n, ev_real.data(), ev_imag.data(), eigenvectors.data(), n, workspace.data(), ws);
    }
    catch (const std::exception& ex)
    {
        std::printf("  [general/with_vectors n=%d] skipped: %s\n", static_cast<int>(n), ex.what());
    }
}

// ---------------------------------------------------------------------------
// Backend sweep: small / medium / large
// ---------------------------------------------------------------------------

template <linalg::linalg_backend Backend>
void profile_backend(const char* backend_label, linalg_int n)
{
    PROFILER_SCOPE(backend_label);

    char scope_name[64];

    std::snprintf(scope_name, sizeof(scope_name), "n=%d", static_cast<int>(n / 2));
    {
        PROFILER_SCOPE(scope_name);
        profile_eigen_at_size<Backend>(n / 2 > 0 ? n / 2 : 1);
    }

    std::snprintf(scope_name, sizeof(scope_name), "n=%d", static_cast<int>(n));
    {
        PROFILER_SCOPE(scope_name);
        profile_eigen_at_size<Backend>(n);
    }

    std::snprintf(scope_name, sizeof(scope_name), "n=%d", static_cast<int>(2 * n));
    {
        PROFILER_SCOPE(scope_name);
        profile_eigen_at_size<Backend>(2 * n);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    linalg_int n = 128;
    if (argc > 1)
    {
        char*      end    = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed > 4 && parsed <= 2048)
        {
            n = static_cast<linalg_int>(parsed);
        }
    }

    std::printf("=== LinearAlgebra eigen profiler ===\n");
    std::printf("Base matrix order n=%d  (sweep: %d / %d / %d)\n\n",
        static_cast<int>(n), static_cast<int>(n / 2), static_cast<int>(n), static_cast<int>(2 * n));

    profiler::session_options opts;
    opts.activities           = {profiler::activity::cpu};
    opts.statistical_analysis = true;

    profiler::session session(opts);
    if (!session.start())
    {
        std::fprintf(stderr, "warning: profiler session failed to start\n");
    }

    profile_backend<linalg::linalg_backend::NATIVE>("native", n);

#if LINALG_HAS_BLAS && LINALG_HAS_LAPACKE
    profile_backend<linalg::linalg_backend::BLAS_LAPACKE>("blas_lapacke", n);
#endif

#if LINALG_HAS_MKL
    profile_backend<linalg::linalg_backend::MKL>("mkl", n);
#endif

    session.stop();

    const char* trace_path = "profile_linalg_eigen.json";
    if (session.write_trace(trace_path))
    {
        std::printf("Trace written to %s\n", trace_path);
    }

    auto report = session.generate_report();
    if (report)
    {
        std::printf("\n--- Eigen profiler report ---\n");
        report->print_detailed_report();
    }

    std::printf("\nVisualization: open %s in chrome://tracing or https://ui.perfetto.dev\n", trace_path);
    return 0;
}
