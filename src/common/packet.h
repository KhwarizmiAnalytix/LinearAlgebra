#pragma once

#include <cstddef>

#include "include/common/types.h"

// LinearAlgebra — SIMD packet abstraction for the scalar backend's blocked
// GEMM (see the LINALG_HAS_VECTORIZATION block in multiplication.cpp, an
// Eigen-derived GEBP-style micro-kernel). `linalg::packet<T>` mirrors the
// subset of Eigen's internal packet-math primitives (Eigen/src/Core/arch/
// */PacketMath.h, GenericPacketMathFunctions.h) that kernel actually calls:
// zero-cost on an unsupported platform (falls back to T itself, packet
// width 1) and real vector instructions on NEON/SSE2.
//
// Naming and calling convention (out-parameter style: `op(ins..., out&)`
// rather than `out = op(ins...)`) intentionally match the call sites in
// multiplication.cpp rather than following a more idiomatic
// return-based API — this header exists to satisfy that already-written
// kernel, not the other way around.
//
// LINALG_ASM_COMMENT is an inline-asm no-op Eigen itself uses only to stop
// some compilers reordering across it (worked around a specific old GCC/
// Clang codegen bug); portable no-op here since none of the compilers this
// repository targets need the workaround.
#define LINALG_ASM_COMMENT(msg) ((void)0)

// ISA selection. CMakeLists.txt's LINALG_VECTORIZATION_ISA cache variable
// always has a concrete value — never "auto" — defaulting per
// CMAKE_SYSTEM_PROCESSOR (avx2 on x86_64/AMD64/i386/i686, neon on
// aarch64/arm64, scalar otherwise) when not overridden on the command line.
// Whenever LINALG_ENABLE_VECTORIZATION is ON, CMake defines exactly one of
// the LINALG_SIMD_* macros below unconditionally (and adds the matching
// -m<isa> compiler flag on x86, so the intrinsics below actually assemble).
// There is no compiler-predefined-macro auto-detection fallback any more: if
// none of the six is defined, this header is being built outside that
// CMake-driven path (e.g. standalone, or with LINALG_HAS_VECTORIZATION off
// but packet.h still included) and must fail loudly rather than silently
// guess an ISA.
#if !defined(LINALG_SIMD_SCALAR) && !defined(LINALG_SIMD_SSE2) && !defined(LINALG_SIMD_AVX) && \
    !defined(LINALG_SIMD_AVX2) && !defined(LINALG_SIMD_AVX512) && !defined(LINALG_SIMD_NEON)
#error \
    "packet.h requires exactly one LINALG_SIMD_* macro to be predefined (set by CMakeLists.txt's LINALG_VECTORIZATION_ISA option); define one explicitly if building this header outside that CMake path."
#endif

namespace linalg
{

template <typename T>
struct packet;

// Shared by every packet<T>::ptranspose<N> below (scalar and every ISA
// specialization alike) instead of each duplicating its own shuffle logic.
//
// Two call shapes reach this, both from multiplication.cpp: pack_lhs
// calls ptranspose<PacketSize>, i.e. N == packet<T>::size — an ordinary NxN
// transpose of N whole registers. pack_rhs always calls ptranspose<4> (nr,
// the GEBP micro-kernel's fixed column-block width, independent of the
// vector ISA) regardless of PacketSize — for a packet wider than 4 lanes,
// Eigen's real ptranspose<4> does *not* leave the upper lanes untouched; it
// treats the register as `size / N` independent N-lane groups and
// transposes each group separately using the same N registers, so every
// lane of all N registers ends up holding transposed data. This generic
// version implements that directly (and happens to degenerate to the
// simple NxN case when N == size, since there is then exactly one group),
// trading the per-ISA shuffle instructions a native implementation would
// use for portable, easy-to-verify storeu/loadu round-trips through a
// scratch buffer — correctness over the last bit of performance, since
// this is still outside any hot inner loop.
template <typename T, int N>
void ptranspose_generic(typename packet<T>::packet_t* m)
{
    using P                = packet<T>;
    constexpr int width    = P::size;
    constexpr int n_groups = width / N;
    static_assert(width % N == 0, "ptranspose_generic requires N to evenly divide packet<T>::size");

    T buf[N][width];
    for (int r = 0; r < N; ++r)
    {
        P::storeu(m[r], buf[r]);
    }

    T out[N][width];
    for (int g = 0; g < n_groups; ++g)
    {
        const int base = g * N;
        for (int r = 0; r < N; ++r)
        {
            for (int c = 0; c < N; ++c)
            {
                out[r][base + c] = buf[c][base + r];
            }
        }
    }

    for (int r = 0; r < N; ++r)
    {
        P::loadu(out[r], m[r]);
    }
}

// Shared by every packet<T>::predux below (scalar and every ISA
// specialization alike), the same way ptranspose_generic is shared for
// transposes above: a horizontal sum of all lanes, used by
// cholesky_decomposition.cpp's blocked trailing-submatrix update to reduce
// a dot-product accumulator to a scalar. Trading the per-ISA horizontal-add
// instructions a native implementation would use (haddps/vaddvq_f32/...)
// for a portable, easy-to-verify storeu-then-sum — correctness over the
// last bit of performance, exactly like ptranspose_generic's rationale.
template <typename T>
T predux_generic(typename packet<T>::packet_t v)
{
    T buf[packet<T>::size];
    packet<T>::storeu(v, buf);
    T sum = buf[0];
    for (int i = 1; i < packet<T>::size; ++i)
    {
        sum += buf[i];
    }
    return sum;
}

// Primary template: scalar fallback, packet width 1. Correct on every
// platform by construction — every operation below degenerates to the
// plain scalar statement it would be without vectorization at all.
template <typename T>
struct packet
{
    using packet_t      = T;
    using packet_half_t = T;

    static constexpr int size      = 1;
    static constexpr int half_size = 1;

    static void copy(const T* src, T* dst) { *dst = *src; }
    static void load(const T* ptr, packet_t& out) { out = *ptr; }
    static void loadu(const T* ptr, packet_t& out) { out = *ptr; }
    static T    loadu_half(const T* ptr) { return *ptr; }
    static void store(packet_t v, T* ptr) { *ptr = v; }
    static void storeu(packet_t v, T* ptr) { *ptr = v; }
    static void set(T scalar, packet_t& out) { out = scalar; }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out) { out = a * b + c; }
    static void add(packet_t a, packet_t b, packet_t& out) { out = a + b; }
    static void div(packet_t a, packet_t b, packet_t& out) { out = a / b; }
    static void prefetch(const T* /*ptr*/) {}

    static void broadcast(const T* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = ptr[0];
        b1 = ptr[1];
        b2 = ptr[2];
        b3 = ptr[3];
    }

    static void ploadquad(const T* ptr, packet_t& out) { out = *ptr; }
    static packet_t predux_downto4(packet_t v) { return v; }
    static T        predux(packet_t v) { return v; }

    static void gather(const T* ptr, linalg_int stride, packet_t& out) { out = *ptr; (void)stride; }
    static void scatter(packet_t v, linalg_int stride, T* ptr) { *ptr = v; (void)stride; }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<T, N>(m);
    }
};

}  // namespace linalg

#if defined(LINALG_SIMD_NEON)
#include <arm_neon.h>

namespace linalg
{

template <>
struct packet<float>
{
    using packet_t      = float32x4_t;
    using packet_half_t = float32x2_t;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    static void copy(const float* src, float* dst) { vst1q_f32(dst, vld1q_f32(src)); }
    static void load(const float* ptr, packet_t& out) { out = vld1q_f32(ptr); }
    static void loadu(const float* ptr, packet_t& out) { out = vld1q_f32(ptr); }
    static float loadu_half(const float* ptr) { return vget_lane_f32(vld1_f32(ptr), 0); }
    static void store(packet_t v, float* ptr) { vst1q_f32(ptr, v); }
    static void storeu(packet_t v, float* ptr) { vst1q_f32(ptr, v); }
    static void set(float scalar, packet_t& out) { out = vdupq_n_f32(scalar); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out) { out = vfmaq_f32(c, a, b); }
    static void add(packet_t a, packet_t b, packet_t& out) { out = vaddq_f32(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = vdivq_f32(a, b); }
    static void prefetch(const float* /*ptr*/) {}

    static void broadcast(const float* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = vdupq_n_f32(ptr[0]);
        b1 = vdupq_n_f32(ptr[1]);
        b2 = vdupq_n_f32(ptr[2]);
        b3 = vdupq_n_f32(ptr[3]);
    }

    // Reads one scalar and broadcasts it across the packet — same shape as
    // `set`, used by the "process remaining rows, 1 at once" loop below
    // where the single-row operand is walked `spk` (here: 1) element at a
    // time rather than a full packet at a time.
    static void ploadquad(const float* ptr, packet_t& out) { out = vdupq_n_f32(*ptr); }

    // Only ever called when size > 4 (AVX/AVX-512-width packets); for a
    // native 4-wide packet it is already "downto 4", i.e. the identity.
    static packet_t predux_downto4(packet_t v) { return v; }
    static float    predux(packet_t v) { return predux_generic<float>(v); }

    static void gather(const float* ptr, linalg_int stride, packet_t& out)
    {
        alignas(16) float lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = vld1q_f32(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, float* ptr)
    {
        alignas(16) float lanes[4];
        vst1q_f32(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<float, N>(m);
    }
};

template <>
struct packet<double>
{
    using packet_t      = float64x2_t;
    using packet_half_t = double;

    static constexpr int size      = 2;
    static constexpr int half_size = 1;

    static void copy(const double* src, double* dst) { vst1q_f64(dst, vld1q_f64(src)); }
    static void load(const double* ptr, packet_t& out) { out = vld1q_f64(ptr); }
    static void loadu(const double* ptr, packet_t& out) { out = vld1q_f64(ptr); }
    static double loadu_half(const double* ptr) { return *ptr; }
    static void store(packet_t v, double* ptr) { vst1q_f64(ptr, v); }
    static void storeu(packet_t v, double* ptr) { vst1q_f64(ptr, v); }
    static void set(double scalar, packet_t& out) { out = vdupq_n_f64(scalar); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out) { out = vfmaq_f64(c, a, b); }
    static void add(packet_t a, packet_t b, packet_t& out) { out = vaddq_f64(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = vdivq_f64(a, b); }
    static void prefetch(const double* /*ptr*/) {}

    static void broadcast(const double* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = vdupq_n_f64(ptr[0]);
        b1 = vdupq_n_f64(ptr[1]);
        b2 = vdupq_n_f64(ptr[2]);
        b3 = vdupq_n_f64(ptr[3]);
    }

    static void ploadquad(const double* ptr, packet_t& out) { out = vdupq_n_f64(*ptr); }

    static packet_t predux_downto4(packet_t v) { return v; }
    static double   predux(packet_t v) { return predux_generic<double>(v); }

    static void gather(const double* ptr, linalg_int stride, packet_t& out)
    {
        alignas(16) double lanes[2];
        lanes[0] = ptr[0];
        lanes[1] = ptr[stride];
        out      = vld1q_f64(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, double* ptr)
    {
        alignas(16) double lanes[2];
        vst1q_f64(lanes, v);
        ptr[0]      = lanes[0];
        ptr[stride] = lanes[1];
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<double, N>(m);
    }
};

}  // namespace linalg

#elif defined(LINALG_SIMD_SSE2)
#include <emmintrin.h>

// Baseline SSE2 (always available on x86-64) has no fused multiply-add
// instruction, so `fma` here is a separate mul + add — the same tradeoff
// the scalar fallback above makes, not a regression relative to not
// vectorizing at all. Written following standard, well-documented SSE2
// intrinsic patterns; unlike the NEON specializations above, this has not
// been run on real x86 hardware as part of this change.
namespace linalg
{

template <>
struct packet<float>
{
    using packet_t      = __m128;
    using packet_half_t = __m64;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    static void copy(const float* src, float* dst) { _mm_storeu_ps(dst, _mm_loadu_ps(src)); }
    static void load(const float* ptr, packet_t& out) { out = _mm_loadu_ps(ptr); }
    static void loadu(const float* ptr, packet_t& out) { out = _mm_loadu_ps(ptr); }
    static float loadu_half(const float* ptr) { return ptr[0]; }
    static void store(packet_t v, float* ptr) { _mm_storeu_ps(ptr, v); }
    static void storeu(packet_t v, float* ptr) { _mm_storeu_ps(ptr, v); }
    static void set(float scalar, packet_t& out) { out = _mm_set1_ps(scalar); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out)
    {
        out = _mm_add_ps(_mm_mul_ps(a, b), c);
    }
    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm_add_ps(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm_div_ps(a, b); }
    static void prefetch(const float* /*ptr*/) {}

    static void broadcast(const float* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm_set1_ps(ptr[0]);
        b1 = _mm_set1_ps(ptr[1]);
        b2 = _mm_set1_ps(ptr[2]);
        b3 = _mm_set1_ps(ptr[3]);
    }

    static void ploadquad(const float* ptr, packet_t& out) { out = _mm_set1_ps(*ptr); }

    static packet_t predux_downto4(packet_t v) { return v; }
    static float    predux(packet_t v) { return predux_generic<float>(v); }

    static void gather(const float* ptr, linalg_int stride, packet_t& out)
    {
        alignas(16) float lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm_loadu_ps(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, float* ptr)
    {
        alignas(16) float lanes[4];
        _mm_storeu_ps(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<float, N>(m);
    }
};

template <>
struct packet<double>
{
    using packet_t      = __m128d;
    using packet_half_t = double;

    static constexpr int size      = 2;
    static constexpr int half_size = 1;

    static void copy(const double* src, double* dst) { _mm_storeu_pd(dst, _mm_loadu_pd(src)); }
    static void load(const double* ptr, packet_t& out) { out = _mm_loadu_pd(ptr); }
    static void loadu(const double* ptr, packet_t& out) { out = _mm_loadu_pd(ptr); }
    static double loadu_half(const double* ptr) { return ptr[0]; }
    static void store(packet_t v, double* ptr) { _mm_storeu_pd(ptr, v); }
    static void storeu(packet_t v, double* ptr) { _mm_storeu_pd(ptr, v); }
    static void set(double scalar, packet_t& out) { out = _mm_set1_pd(scalar); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out)
    {
        out = _mm_add_pd(_mm_mul_pd(a, b), c);
    }
    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm_add_pd(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm_div_pd(a, b); }
    static void prefetch(const double* /*ptr*/) {}

    static void broadcast(const double* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm_set1_pd(ptr[0]);
        b1 = _mm_set1_pd(ptr[1]);
        b2 = _mm_set1_pd(ptr[2]);
        b3 = _mm_set1_pd(ptr[3]);
    }

    static void ploadquad(const double* ptr, packet_t& out) { out = _mm_set1_pd(*ptr); }

    static packet_t predux_downto4(packet_t v) { return v; }
    static double   predux(packet_t v) { return predux_generic<double>(v); }

    static void gather(const double* ptr, linalg_int stride, packet_t& out)
    {
        alignas(16) double lanes[2];
        lanes[0] = ptr[0];
        lanes[1] = ptr[stride];
        out      = _mm_loadu_pd(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, double* ptr)
    {
        alignas(16) double lanes[2];
        _mm_storeu_pd(lanes, v);
        ptr[0]      = lanes[0];
        ptr[stride] = lanes[1];
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<double, N>(m);
    }
};

}  // namespace linalg

#elif defined(LINALG_SIMD_AVX) || defined(LINALG_SIMD_AVX2)
#include <immintrin.h>

// AVX and AVX2 share the same 256-bit register types and most intrinsics;
// they differ only in `fma` (AVX2 is normally paired with -mfma, giving a
// real fused multiply-add; plain AVX has none, so it falls back to a
// separate multiply + add, same tradeoff as the SSE2 specialization above)
// and in `half_size`/the "process remaining rows, 1 at once" fallback
// below. Written following standard, well-documented AVX/AVX2 intrinsic
// patterns; like the SSE2 specialization above, this has not been run on
// real x86 hardware as part of this change.
namespace linalg
{

template <>
struct packet<float>
{
    using packet_t      = __m256;
    using packet_half_t = __m128;

    static constexpr int size = 8;

    // Deliberately not half_size's mathematically honest value (4): the
    // only reader of half_size is multiplication.cpp's "process
    // remaining rows, 1 at once" loop, which — for a packet this wide —
    // asks packet<T>::ploadquad/predux_downto4/gather/scatter to do more
    // than broadcast-and-reduce a single scalar (Eigen's own pload_quad
    // widens its meaning past packet width 4), and this header does not
    // implement that wider semantic. Forcing half_size away from `nr` (4)
    // keeps that loop's `if constexpr` on its other, verified branch (the
    // plain scalar fallback for the row remainder) instead of silently
    // computing a wrong answer for a semantic this header does not
    // implement. The 3x/2x/1x-LhsProgress blocked loops that make up the
    // bulk of the GEMM — this specialization's actual payoff — do not
    // depend on half_size at all.
    static constexpr int half_size = -1;

    static void copy(const float* src, float* dst) { _mm256_storeu_ps(dst, _mm256_loadu_ps(src)); }
    static void load(const float* ptr, packet_t& out) { out = _mm256_loadu_ps(ptr); }
    static void loadu(const float* ptr, packet_t& out) { out = _mm256_loadu_ps(ptr); }
    static packet_half_t loadu_half(const float* ptr) { return _mm_loadu_ps(ptr); }
    static void store(packet_t v, float* ptr) { _mm256_storeu_ps(ptr, v); }
    static void storeu(packet_t v, float* ptr) { _mm256_storeu_ps(ptr, v); }
    static void set(float scalar, packet_t& out) { out = _mm256_set1_ps(scalar); }
    static packet_half_t set(const float* ptr) { return _mm_set1_ps(*ptr); }

    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out)
    {
#if defined(LINALG_SIMD_AVX2)
        out = _mm256_fmadd_ps(a, b, c);
#else
        out = _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
    }

    static void fma(packet_half_t a, packet_half_t b, packet_half_t& c)
    {
#if defined(LINALG_SIMD_AVX2)
        c = _mm_fmadd_ps(a, b, c);
#else
        c = _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
    }

    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm256_add_ps(a, b); }
    static void add(packet_half_t a, packet_half_t b, packet_half_t& out) { out = _mm_add_ps(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm256_div_ps(a, b); }
    static void prefetch(const float* /*ptr*/) {}

    static void broadcast(const float* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm256_set1_ps(ptr[0]);
        b1 = _mm256_set1_ps(ptr[1]);
        b2 = _mm256_set1_ps(ptr[2]);
        b3 = _mm256_set1_ps(ptr[3]);
    }

    static void ploadquad(const float* ptr, packet_t& out) { out = _mm256_set1_ps(*ptr); }

    // Reduces an 8-wide packet to 4-wide by adding its low and high 128-bit
    // halves — dead code while half_size (above) keeps the one call site
    // that would use this on its scalar-fallback branch; implemented
    // honestly in case that is ever revisited.
    static packet_half_t predux_downto4(packet_t v)
    {
        return _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    }

    static float predux(packet_t v) { return predux_generic<float>(v); }

    static void gather(const float* ptr, linalg_int stride, packet_t& out)
    {
        alignas(32) float lanes[8];
        for (int i = 0; i < 8; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm256_loadu_ps(lanes);
    }

    static void gather(const float* ptr, linalg_int stride, packet_half_t& out)
    {
        alignas(16) float lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm_loadu_ps(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, float* ptr)
    {
        alignas(32) float lanes[8];
        _mm256_storeu_ps(lanes, v);
        for (int i = 0; i < 8; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    static void scatter(packet_half_t v, linalg_int stride, float* ptr)
    {
        alignas(16) float lanes[4];
        _mm_storeu_ps(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<float, N>(m);
    }
};

template <>
struct packet<double>
{
    using packet_t      = __m256d;
    using packet_half_t = __m128d;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    static void copy(const double* src, double* dst) { _mm256_storeu_pd(dst, _mm256_loadu_pd(src)); }
    static void load(const double* ptr, packet_t& out) { out = _mm256_loadu_pd(ptr); }
    static void loadu(const double* ptr, packet_t& out) { out = _mm256_loadu_pd(ptr); }
    static double loadu_half(const double* ptr) { return ptr[0]; }
    static void store(packet_t v, double* ptr) { _mm256_storeu_pd(ptr, v); }
    static void storeu(packet_t v, double* ptr) { _mm256_storeu_pd(ptr, v); }
    static void set(double scalar, packet_t& out) { out = _mm256_set1_pd(scalar); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out)
    {
#if defined(LINALG_SIMD_AVX2)
        out = _mm256_fmadd_pd(a, b, c);
#else
        out = _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
    }
    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm256_add_pd(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm256_div_pd(a, b); }
    static void prefetch(const double* /*ptr*/) {}

    static void broadcast(const double* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm256_set1_pd(ptr[0]);
        b1 = _mm256_set1_pd(ptr[1]);
        b2 = _mm256_set1_pd(ptr[2]);
        b3 = _mm256_set1_pd(ptr[3]);
    }

    static void ploadquad(const double* ptr, packet_t& out) { out = _mm256_set1_pd(*ptr); }

    static packet_t predux_downto4(packet_t v) { return v; }
    static double   predux(packet_t v) { return predux_generic<double>(v); }

    static void gather(const double* ptr, linalg_int stride, packet_t& out)
    {
        alignas(32) double lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm256_loadu_pd(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, double* ptr)
    {
        alignas(32) double lanes[4];
        _mm256_storeu_pd(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<double, N>(m);
    }
};

}  // namespace linalg

#elif defined(LINALG_SIMD_AVX512)
#include <immintrin.h>

// Unlike plain AVX, AVX-512F always includes fused multiply-add. Written
// following standard, well-documented AVX-512 intrinsic patterns; like the
// SSE2/AVX specializations above, this has not been run on real x86
// hardware as part of this change.
namespace linalg
{

template <>
struct packet<float>
{
    using packet_t      = __m512;
    using packet_half_t = __m128;

    static constexpr int size = 16;

    // Always unreachable regardless of value: multiplication.cpp's
    // "process remaining rows, 1 at once" loop only consults half_size
    // when LhsProgress <= 8, and this packet's LhsProgress (== size) is
    // 16 — so this is never compared against anything. Kept at the
    // mathematically honest value for documentation purposes only.
    static constexpr int half_size = 4;

    static void copy(const float* src, float* dst) { _mm512_storeu_ps(dst, _mm512_loadu_ps(src)); }
    static void load(const float* ptr, packet_t& out) { out = _mm512_loadu_ps(ptr); }
    static void loadu(const float* ptr, packet_t& out) { out = _mm512_loadu_ps(ptr); }
    static packet_half_t loadu_half(const float* ptr) { return _mm_loadu_ps(ptr); }
    static void store(packet_t v, float* ptr) { _mm512_storeu_ps(ptr, v); }
    static void storeu(packet_t v, float* ptr) { _mm512_storeu_ps(ptr, v); }
    static void set(float scalar, packet_t& out) { out = _mm512_set1_ps(scalar); }
    static packet_half_t set(const float* ptr) { return _mm_set1_ps(*ptr); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out) { out = _mm512_fmadd_ps(a, b, c); }
    static void fma(packet_half_t a, packet_half_t b, packet_half_t& c) { c = _mm_fmadd_ps(a, b, c); }
    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm512_add_ps(a, b); }
    static void add(packet_half_t a, packet_half_t b, packet_half_t& out) { out = _mm_add_ps(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm512_div_ps(a, b); }
    static void prefetch(const float* /*ptr*/) {}

    static void broadcast(const float* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm512_set1_ps(ptr[0]);
        b1 = _mm512_set1_ps(ptr[1]);
        b2 = _mm512_set1_ps(ptr[2]);
        b3 = _mm512_set1_ps(ptr[3]);
    }

    static void ploadquad(const float* ptr, packet_t& out) { out = _mm512_set1_ps(*ptr); }

    // Reduces 16-wide to 4-wide: fold to 8-wide (low + high 256-bit half),
    // then to 4-wide (low + high 128-bit half of that) — dead code, see
    // half_size above; implemented honestly in case that changes.
    static packet_half_t predux_downto4(packet_t v)
    {
        const __m256 lo8 = _mm512_castps512_ps256(v);
        const __m256 hi8 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        const __m256 s8  = _mm256_add_ps(lo8, hi8);
        return _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    }

    static float predux(packet_t v) { return predux_generic<float>(v); }

    static void gather(const float* ptr, linalg_int stride, packet_t& out)
    {
        alignas(64) float lanes[16];
        for (int i = 0; i < 16; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm512_loadu_ps(lanes);
    }

    static void gather(const float* ptr, linalg_int stride, packet_half_t& out)
    {
        alignas(16) float lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm_loadu_ps(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, float* ptr)
    {
        alignas(64) float lanes[16];
        _mm512_storeu_ps(lanes, v);
        for (int i = 0; i < 16; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    static void scatter(packet_half_t v, linalg_int stride, float* ptr)
    {
        alignas(16) float lanes[4];
        _mm_storeu_ps(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<float, N>(m);
    }
};

template <>
struct packet<double>
{
    using packet_t      = __m512d;
    using packet_half_t = __m256d;

    static constexpr int size = 8;

    // See packet<float>::half_size (AVX/AVX2 specialization above) for why
    // this is deliberately not the honest value (4): this packet's
    // LhsProgress (== size) is 8, so multiplication.cpp's "process
    // remaining rows, 1 at once" loop would otherwise take the branch this
    // header does not implement a verified semantic for.
    static constexpr int half_size = -1;

    static void copy(const double* src, double* dst) { _mm512_storeu_pd(dst, _mm512_loadu_pd(src)); }
    static void load(const double* ptr, packet_t& out) { out = _mm512_loadu_pd(ptr); }
    static void loadu(const double* ptr, packet_t& out) { out = _mm512_loadu_pd(ptr); }
    static packet_half_t loadu_half(const double* ptr) { return _mm256_loadu_pd(ptr); }
    static void store(packet_t v, double* ptr) { _mm512_storeu_pd(ptr, v); }
    static void storeu(packet_t v, double* ptr) { _mm512_storeu_pd(ptr, v); }
    static void set(double scalar, packet_t& out) { out = _mm512_set1_pd(scalar); }
    static packet_half_t set(const double* ptr) { return _mm256_set1_pd(*ptr); }
    static void fma(packet_t a, packet_t b, packet_t c, packet_t& out) { out = _mm512_fmadd_pd(a, b, c); }
    static void fma(packet_half_t a, packet_half_t b, packet_half_t& c) { c = _mm256_fmadd_pd(a, b, c); }
    static void add(packet_t a, packet_t b, packet_t& out) { out = _mm512_add_pd(a, b); }
    static void add(packet_half_t a, packet_half_t b, packet_half_t& out) { out = _mm256_add_pd(a, b); }
    static void div(packet_t a, packet_t b, packet_t& out) { out = _mm512_div_pd(a, b); }
    static void prefetch(const double* /*ptr*/) {}

    static void broadcast(const double* ptr, packet_t& b0, packet_t& b1, packet_t& b2, packet_t& b3)
    {
        b0 = _mm512_set1_pd(ptr[0]);
        b1 = _mm512_set1_pd(ptr[1]);
        b2 = _mm512_set1_pd(ptr[2]);
        b3 = _mm512_set1_pd(ptr[3]);
    }

    static void ploadquad(const double* ptr, packet_t& out) { out = _mm512_set1_pd(*ptr); }

    // Reduces 8-wide to 4-wide by adding the low and high 256-bit halves —
    // dead code while half_size (above) keeps the one call site that would
    // use this on its scalar-fallback branch; implemented honestly in case
    // that is ever revisited.
    static packet_half_t predux_downto4(packet_t v)
    {
        return _mm256_add_pd(
            _mm512_castpd512_pd256(v), _mm512_extractf64x4_pd(v, 1));
    }

    static double predux(packet_t v) { return predux_generic<double>(v); }

    static void gather(const double* ptr, linalg_int stride, packet_t& out)
    {
        alignas(64) double lanes[8];
        for (int i = 0; i < 8; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm512_loadu_pd(lanes);
    }

    static void gather(const double* ptr, linalg_int stride, packet_half_t& out)
    {
        alignas(32) double lanes[4];
        for (int i = 0; i < 4; ++i)
        {
            lanes[i] = ptr[i * stride];
        }
        out = _mm256_loadu_pd(lanes);
    }

    static void scatter(packet_t v, linalg_int stride, double* ptr)
    {
        alignas(64) double lanes[8];
        _mm512_storeu_pd(lanes, v);
        for (int i = 0; i < 8; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    static void scatter(packet_half_t v, linalg_int stride, double* ptr)
    {
        alignas(32) double lanes[4];
        _mm256_storeu_pd(lanes, v);
        for (int i = 0; i < 4; ++i)
        {
            ptr[i * stride] = lanes[i];
        }
    }

    template <int N>
    static void ptranspose(packet_t* m)
    {
        ptranspose_generic<double, N>(m);
    }
};

}  // namespace linalg

#endif
