// Eigen adaptive inverse-scaling / Gauss-Legendre matrix-logarithm port.
// Adapted from Eigen unsupported/Eigen/src/MatrixFunctions/MatrixLogarithm.h.
// Assumes T is float or double; matrix order is row-major.
// Original adaptation: Eigen MatrixLogarithm.h (MPL-2.0).
// Preserve the upstream MPL-2.0 attribution and license when distributing.
// No standard-library includes: this file may be stringified for Metal.
#include "src/common/square_root_core.h"

// clang-format off
LINALG_LOGARITHM_CORE(
namespace linalg_schur {

template <typename T>
LINALG_INLINE scalars::complex<T> logarithm(scalars::complex<T> a) {
    return scalars::complex<T>(log(scalars::abs(a)), atan2(a.imag(), a.real()));
}

// 18-term Taylor evaluation for |x| < 1/2; no platform-dependent expm1.
// Relative truncation error is well below double epsilon on this interval.
template <typename T>
LINALG_INLINE T real_exponential_minus_one(T x) {
    if (fabs(x) >= T(0.5)) return exp(x) - T(1);
    T p = T(1);
    for (int k = 18; k >= 2; --k) {
        p = T(1) + x * p / T(k);
    }
    return x * p;
}

template <typename T>
LINALG_INLINE scalars::complex<T> exponential_minus_one(scalars::complex<T> a) {
    const T e = exp(a.real());
    const T sh = sin(a.imag() * T(0.5));
    const T re = real_exponential_minus_one(a.real());
    return scalars::complex<T>(re - T(2) * e * sh * sh, e * sin(a.imag()));
}

// Eigen's degree cutoffs, expressed for the two supported real precisions.
// epsilon supplied by the caller is used only to select float versus double.
template <typename T>
LINALG_INLINE int logarithm_degree(T norm, T epsilon) {
    if (epsilon > T(1.0e-10)) {
        if (norm <= T(0.25111573934555054)) return 3;
        if (norm <= T(0.40535837411880493)) return 4;
        return 5;
    }
    if (norm <= T(0.016206284795015624)) return 3;
    if (norm <= T(0.053873532631381171)) return 4;
    if (norm <= T(0.11352802267628681)) return 5;
    if (norm <= T(0.18662860613541288)) return 6;
    return 7;
}

template <typename T>
LINALG_INLINE T logarithm_max_norm(T epsilon) {
    return epsilon > T(1.0e-10) ? T(0.53149729967117310)
                               : T(0.2642960831111435);
}

// Rows correspond to Gauss-Legendre degrees 3,4,5,6,7.
// The shared host pipeline always launches seven solves; stages >= degree
// are packed as identity/no-op to preserve the original interface.
template <typename T>
LINALG_INLINE T logarithm_node(int degree, int index) {
    const T nodes[5][7] = {
        {T(0.11270166537925831),T(0.5),T(0.88729833462074169),T(0),T(0),T(0),T(0)},
        {T(0.06943184420297371),T(0.33000947820757187),T(0.6699905217924281),T(0.9305681557970263),T(0),T(0),T(0)},
        {T(0.046910077030668),T(0.23076534494715845),T(0.5),T(0.7692346550528415),T(0.953089922969332),T(0),T(0)},
        {T(0.03376524289842399),T(0.16939530676686774),T(0.38069040695840155),T(0.6193095930415985),T(0.8306046932331323),T(0.966234757101576),T(0)},
        {T(0.0254460438286207377),T(0.1292344072003027801),T(0.2970774243113014165),T(0.5),T(0.7029225756886985835),T(0.8707655927996972199),T(0.9745539561713792623)}
    };
    return nodes[degree-3][index];
}

template <typename T>
LINALG_INLINE T logarithm_weight(int degree, int index) {
    const T weights[5][7] = {
        {T(0.27777777777777778),T(0.44444444444444444),T(0.27777777777777778),T(0),T(0),T(0),T(0)},
        {T(0.17392742256872693),T(0.32607257743127307),T(0.32607257743127307),T(0.17392742256872693),T(0),T(0),T(0)},
        {T(0.11846344252809454),T(0.23931433524968323),T(0.28444444444444444),T(0.23931433524968323),T(0.11846344252809454),T(0),T(0)},
        {T(0.08566224618958517),T(0.1803807865240693),T(0.23395696728634552),T(0.23395696728634552),T(0.1803807865240693),T(0.08566224618958517),T(0)},
        {T(0.0647424830844348466),T(0.1398526957446383340),T(0.1909150252525594725),T(0.2089795918367346939),T(0.1909150252525594725),T(0.1398526957446383340),T(0.0647424830844348466)}
    };
    return weights[degree-3][index];
}

template <typename T>
LINALG_INLINE bool prepare_triangular_logarithm(
    LINALG_ADDRESS scalars::complex<T>* __restrict h,
    LINALG_ADDRESS scalars::complex<T>* __restrict f,
    LINALG_ADDRESS scalars::complex<T>* __restrict diagonal,
    int n, T epsilon, LINALG_ADDRESS T* __restrict state) {
    for (int i = 0; i < n; ++i) {
        const auto* row = h + i * n;
        diagonal[i] = logarithm(row[i]);
    }

    int roots = 0;
    T multiplier = T(1);
    int extra_roots = 0;
    int degree = 7;
    for (;;) {
        T norm = T(0);
        for (int j = 0; j < n; ++j) {
            f[j] = scalars::complex<T>(T(0));
        }
        for (int i = 0; i < n; ++i) {
            const auto* row = h + i * n;
            for (int j = i + 1; j < n; ++j) {
                f[j] = f[j] + scalars::complex<T>(scalars::abs(row[j]));
            }
            const T column = f[i].real() + scalars::abs(row[i] - scalars::complex<T>(T(1)));
            norm = fmax(norm, column);
        }
        // Eigen: a second root may be cheaper than a higher Padé degree.
        if (!(norm >= T(0))) return false; // includes NaN
        if (norm < logarithm_max_norm(epsilon)) {
            degree = logarithm_degree(norm, epsilon);
            const int degree2 = logarithm_degree(norm / T(2), epsilon);
            if (degree - degree2 <= 1 || extra_roots == 1) break;
            ++extra_roots;
        }
        if (++roots > 64 || !triangular_sqrt(h, n)) return false;
        multiplier *= T(2);
    }

    const T inv_multiplier = T(1) / multiplier;
    for (int i = 0; i < n; ++i) {
        auto* row = h + i * n;
        row[i] = exponential_minus_one(diagonal[i] * inv_multiplier);
    }
    auto* f_end = f + n * n;
    for (auto* entry = f; entry != f_end; ++entry) {
        *entry = scalars::complex<T>(T(0));
    }
    state[0] = T(degree); // positive success code also stores Padé degree
    state[1] = multiplier;
    return true;
}

template <typename T>
LINALG_INLINE bool logarithm_triangular(
    LINALG_ADDRESS scalars::complex<T>* __restrict h,
    LINALG_ADDRESS scalars::complex<T>* __restrict f,
    LINALG_ADDRESS scalars::complex<T>* __restrict diagonal,
    int n, T epsilon, LINALG_ADDRESS T* __restrict state) {
    for (int i = 0; i < n; ++i) {
        const auto* row = h + i * n;
        const auto lambda = row[i];
        if (scalars::abs(lambda) == T(0) ||
            (lambda.real() < T(0) &&
             fabs(lambda.imag()) <= T(64) * epsilon * fabs(lambda.real())))
            return false;
    }
    return prepare_triangular_logarithm(h, f, diagonal, n, epsilon, state);
}

template <typename T>
LINALG_INLINE void logarithm_schur_prepare(
    LINALG_ADDRESS const T* __restrict sr,
    LINALG_ADDRESS const T* __restrict si, int n,
    LINALG_ADDRESS T* __restrict h_buffer,
    LINALG_ADDRESS T* __restrict f_buffer,
    LINALG_ADDRESS T* __restrict diagonal_buffer,
    LINALG_ADDRESS T* __restrict state,
    T epsilon) {
    auto h = reinterpret_cast<LINALG_ADDRESS scalars::complex<T>*>(h_buffer);
    auto f = reinterpret_cast<LINALG_ADDRESS scalars::complex<T>*>(f_buffer);
    auto diagonal = reinterpret_cast<LINALG_ADDRESS scalars::complex<T>*>(diagonal_buffer);
    unpack_triangular(sr, si, n, h);
    state[0] = T(-1);
    logarithm_triangular(h, f, diagonal, n, epsilon, state);
}

// Same API and real TRSM layout as the original. Each complex element is
// divided just once, not twice; entries below the triangular diagonal are
// zeroed directly without any complex arithmetic.
template <typename T>
LINALG_INLINE void pack_logarithm_solve(
    LINALG_ADDRESS const T* __restrict h_buffer,
    LINALG_ADDRESS const T* __restrict state,
    LINALG_ADDRESS T* __restrict a,
    LINALG_ADDRESS T* __restrict b, int n, int node) {
    auto h = reinterpret_cast<LINALG_ADDRESS const scalars::complex<T>*>(h_buffer);
    const int degree = state[0] > T(0) ? int(state[0]) : 7;
    const bool active = state[0] > T(0) && node < degree;
    const T t = active ? logarithm_node<T>(degree, node) : T(0);
    const int order = 2 * n;
    const bool success = state[0] > T(0);
    const T failure = T(NAN);

    for (int i = 0; i < n; ++i) {
        const auto* h_row = h + i * n;
        const auto d = success ? scalars::complex<T>(T(1)) + h_row[i] * t : scalars::complex<T>(T(1));
        const int row0 = (2*i) * order;
        const int row1 = row0 + order;
        const int rhs0 = (2*i) * n;
        const int rhs1 = rhs0 + n;
        auto* a_row0 = a + row0;
        auto* a_row1 = a + row1;
        auto* b_row0 = b + rhs0;
        auto* b_row1 = b + rhs1;
        for (int j = 0; j < n; ++j) {
            auto coeff = scalars::complex<T>(i == j ? T(1) : T(0));
            auto rhs = scalars::complex<T>(success ? T(0) : failure);
            if (active && i <= j) {
                rhs = h_row[j] / d;
                if (i < j) coeff = rhs * t;
            }
            const int k = 2*j;
            a_row0[k]     = coeff.real();
            a_row0[k + 1] = -coeff.imag();
            a_row1[k]     = coeff.imag();
            a_row1[k + 1] = coeff.real();
            b_row0[j] = rhs.real();
            b_row1[j] = rhs.imag();
        }
    }
}

template <typename T>
LINALG_INLINE void accumulate_logarithm_solve(
    LINALG_ADDRESS T* __restrict f_buffer,
    LINALG_ADDRESS const T* __restrict state,
    LINALG_ADDRESS const T* __restrict solution, int n, int node) {
    if (state[0] < T(3) || node >= int(state[0])) return;
    auto f = reinterpret_cast<LINALG_ADDRESS scalars::complex<T>*>(f_buffer);
    const T w = logarithm_weight<T>(int(state[0]), node);
    // The solution should be mathematically upper triangular; do not
    // accumulate tiny lower-triangular BLAS roundoff.
    for (int i = 0; i < n; ++i) {
        const int r = (2*i)*n;
        auto* f_row = f + i * n;
        const auto* solution_real = solution + r;
        const auto* solution_imag = solution_real + n;
        for (int j = i; j < n; ++j) {
            f_row[j] = f_row[j] + scalars::complex<T>(solution_real[j], solution_imag[j]) * w;
        }
    }
}

template <typename T>
LINALG_INLINE void finish_logarithm(
    LINALG_ADDRESS T* __restrict f_buffer,
    LINALG_ADDRESS const T* __restrict state,
    LINALG_ADDRESS const T* __restrict diagonal_buffer, int n) {
    if (state[0] < T(3)) return;
    auto f = reinterpret_cast<LINALG_ADDRESS scalars::complex<T>*>(f_buffer);
    auto diagonal = reinterpret_cast<LINALG_ADDRESS const scalars::complex<T>*>(diagonal_buffer);
    const T multiplier = state[1];
    for (int i = 0; i < n; ++i) {
        auto* row = f + i * n;
        for (int j = i + 1; j < n; ++j) {
            row[j] = row[j] * multiplier;
        }
        row[i] = diagonal[i];
    }
}

template <typename T>
LINALG_INLINE bool logarithm_schur_finish(
    LINALG_ADDRESS T* __restrict f_buffer,
    LINALG_ADDRESS const T* __restrict state,
    LINALG_ADDRESS const T* __restrict diagonal_buffer, int n,
    LINALG_ADDRESS T* __restrict fr, LINALG_ADDRESS T* __restrict fi) {
    finish_logarithm(f_buffer, state, diagonal_buffer, n);
    const bool success = state[0] > T(0);
    auto f = reinterpret_cast<LINALG_ADDRESS const scalars::complex<T>*>(f_buffer);
    pack_result(n, f, fr, fi, success);
    return success;
}
} // namespace linalg_schur
)
// clang-format on
