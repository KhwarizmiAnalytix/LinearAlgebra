#include "include/matrix_operation/matrix_multiplication.h"

#include <include/logging.h>

#if defined(LINALG_ENABLE_MKL)
#include <mkl.h>
#elif defined(LINALG_ENABLE_BLAS)
#ifdef LINALG_BLAS_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#else
#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif  // _MSC_VER
#include <algorithm>
#include <cmath>
#include <limits>

#include "include/memory/allocator.h"
#include "include/util/cpu_info.h"
#ifdef LINALG_VECTORIZED
#include "common/packet.h"
#endif
#endif

namespace linalg
{
namespace detail
{

// Backend implementation is compiled directly into this translation unit,
// gated by the same #if/#elif/#else that also picks which body the public
// matrix_multiplication() overloads below call — exactly one is ever built.
#if defined(LINALG_ENABLE_MKL)
void matmul_mkl_f32(bool transpose_a,
    bool                 transpose_b,
    linalg_int         rows,
    linalg_int         columns,
    linalg_int         depth,
    float const*         a,
    linalg_int         lda,
    float const*         b,
    linalg_int         ldb,
    float*               c,
    linalg_int         ldc)
{
    cblas_sgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.F,
        a,
        lda,
        b,
        ldb,
        0.F,
        c,
        ldc);
}

void matmul_mkl_f64(bool transpose_a,
    bool                 transpose_b,
    linalg_int         rows,
    linalg_int         columns,
    linalg_int         depth,
    double const*        a,
    linalg_int         lda,
    double const*        b,
    linalg_int         ldb,
    double*              c,
    linalg_int         ldc)
{
    cblas_dgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        rows,
        columns,
        depth,
        1.,
        a,
        lda,
        b,
        ldb,
        0.,
        c,
        ldc);
}
#elif defined(LINALG_ENABLE_BLAS)
void matmul_blas_f32(bool transpose_a,
    bool                  transpose_b,
    linalg_int          rows,
    linalg_int          columns,
    linalg_int          depth,
    float const*          a,
    linalg_int          lda,
    float const*          b,
    linalg_int          ldb,
    float*                c,
    linalg_int          ldc)
{
    cblas_sgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        static_cast<int>(rows),
        static_cast<int>(columns),
        static_cast<int>(depth),
        1.F,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        0.F,
        c,
        static_cast<int>(ldc));
}

void matmul_blas_f64(bool transpose_a,
    bool                  transpose_b,
    linalg_int          rows,
    linalg_int          columns,
    linalg_int          depth,
    double const*         a,
    linalg_int          lda,
    double const*         b,
    linalg_int          ldb,
    double*               c,
    linalg_int          ldc)
{
    cblas_dgemm(CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        static_cast<int>(rows),
        static_cast<int>(columns),
        static_cast<int>(depth),
        1.,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        0.,
        c,
        static_cast<int>(ldc));
}
#else

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

namespace
{
template <typename value_t, bool ColMajor = true>
constexpr const value_t& element(
    value_t const* data, linalg_int i, linalg_int j, linalg_int ld)
{
    if constexpr (ColMajor)
    {
        return data[i + j * ld];
    }
    else
    {
        return data[j + i * ld];
    }
}

template <typename value_t, bool ColMajor = true>
constexpr value_t& element(value_t* data, linalg_int i, linalg_int j, linalg_int ld)
{
    if constexpr (ColMajor)
    {
        return data[i + j * ld];
    }
    else
    {
        return data[j + i * ld];
    }
}

#ifdef LINALG_VECTORIZED
std::ptrdiff_t l1 = 0, l2 = 0, l3 = 0, l3_count = 0;

template <typename value_t, bool vectorizable = true> struct gemm_traits
{
    static constexpr linalg_int size = simd<value_t>::size;
    // vectorizable ? simd<value_t>::size : 1;

    static constexpr linalg_int number_of_registers = (2 * sizeof(void*));

    // register block size along the N direction must be 1 or 4
    static constexpr linalg_int nr = 4;

    // register block size along the M direction (currently, this one cannot be modified)
    static constexpr linalg_int mr = 3 * size;
    // vectorizable ? 3 * size : (plain_min<16, number_of_registers>() / 2 / nr) * size;

    static constexpr linalg_int LhsProgress = size;
    static constexpr linalg_int RhsProgress = 1;
};

template <typename value_t, linalg_int KcFactor = 1>
linalg_int blocking_sizes(
    linalg_int& k, linalg_int& m, linalg_int& n, LINALG_UNUSED linalg_int num_threads = 1)
{
    using Traits = gemm_traits<value_t>;

    // Explanations:
    // Let's recall that the product algorithms form mc x kc vertical panels A' on the lhs and
    // kc x nc blocks B' on the rhs. B' has to fit into L2/L3 cache. Moreover, A' is processed
    // per mr x kc horizontal small panels where mr is the blocking size along the m dimension
    // at the register level. This small horizontal panel has to stay within L1 cache.

    // get the cach size L1, L2 and L3
    if (l1 == 0)
    {
        cpu_info::cpuinfo_cach(l1, l2, l3, l3_count);
    }

    {
        // Early return for small problems because the computation below are time consuming for
        // small problems. Perhaps it would make more sense to consider k*n*m?? Note that for very
        // tiny problem, this function should be bypassed anyway because we use the
        // coefficient-based implementation for them.
        if (std::max<linalg_int>({k, m, n}) < 48)
        {
            return static_cast<linalg_int>(l1);
        }

        enum
        {
            k_peeling = 8,
            k_div     = KcFactor * (Traits::mr * sizeof(value_t) + Traits::nr * sizeof(value_t)),
            k_sub     = Traits::mr * Traits::nr * sizeof(value_t)
        };

        // ---- 1st level of blocking on L1, yields kc ----

        // Blocking on the third dimension (i.e., k) is chosen so that an horizontal panel
        // of size mr x kc of the lhs plus a vertical panel of kc x nr of the rhs both fits within
        // L1 cache. We also include a register-level block of the result (mx x nr). (In an ideal
        // world only the lhs panel would stay in L1) Moreover, kc has to be a multiple of 8 to be
        // compatible with loop peeling, leading to a maximum blocking size of:
        const linalg_int max_kc =
            std::max<linalg_int>(((l1 - k_sub) / k_div) & (~(k_peeling - 1)), 1);
        const linalg_int old_k = k;
        if (k > max_kc)
        {
            // We are really blocking on the third dimension:
            // -> reduce blocking size to make sure the last block is as large as possible
            //    while keeping the same number of sweeps over the result.
            k = (k % max_kc) == 0 ? max_kc
                                  : max_kc - k_peeling * ((max_kc - 1 - (k % max_kc)) /
                                                             (k_peeling * (k / max_kc + 1)));

            // eigen_internal_assert(((old_k / k) == (old_k / max_kc)) && "the number of sweeps has
            // to remain the same");
        }

        // ---- 2nd level of blocking on max(L2,L3), yields nc ----
        const linalg_int actual_l2 =
            std::max<linalg_int>(l2, l3 / l3_count);  // NOLINT // == 1.5 MB

        // Here, nc is chosen such that a block of kc x nc of the rhs fit within half of L2.
        // The second half is implicitly reserved to access the result and lhs coefficients.
        // When k<max_kc, then nc can arbitrarily growth. In practice, it seems to be fruitful
        // to limit this growth: we bound nc to growth by a factor x1.5.
        // However, if the entire lhs block fit within L1, then we are not going to block on the
        // rows at all, and it becomes fruitful to keep the packed rhs blocks in L1 if there is
        // enough remaining space.
        linalg_int       max_nc;
        const linalg_int lhs_bytes    = m * k * sizeof(value_t);  // NOLINT
        const linalg_int remaining_l1 = l1 - k_sub - lhs_bytes;
        if (remaining_l1 >= (linalg_int)(Traits::nr * sizeof(value_t)) * k)
        {
            // L1 blocking
            max_nc = remaining_l1 / (k * sizeof(value_t));
        }
        else
        {
            // L2 blocking
            max_nc = (3 * actual_l2) / (2 * 2 * max_kc * sizeof(value_t));  // NOLINT
        }
        // WARNING Below, we assume that Traits::nr is a power of two.
        const auto nc = std::max<linalg_int>(
                            actual_l2 / (static_cast<long long>(2) * k * sizeof(value_t)), max_nc) &
                        (~(Traits::nr - 1));  // NOLINT
        if (n > nc)
        {
            // We are really blocking over the columns:
            // -> reduce blocking size to make sure the last block is as large as possible
            //    while keeping the same number of sweeps over the packed lhs.
            //    Here we allow one more sweep if this gives us a perfect match, thus the commented
            //    "-1"
            n = (n % nc) == 0
                    ? nc
                    : (nc - Traits::nr * ((nc /*-1*/ - (n % nc)) / (Traits::nr * (n / nc + 1))));
        }
        else if (old_k == k)
        {
            // So far, no blocking at all, i.e., kc==k, and nc==n.
            // In this case, let's perform a blocking over the rows such that the packed lhs data is
            // kept in cache L1/L2
            // TODO: part of this blocking strategy is now implemented within the kernel itself, so
            // the L1-based heuristic here should be obsolete.
            linalg_int problem_size = k * n * sizeof(value_t);  // NOLINT
            linalg_int actual_lm    = actual_l2;
            linalg_int max_mc       = m;
            if (problem_size <= 1024)
            {
                // problem is small enough to keep in L1
                // Let's choose m such that lhs's block fit in 1/3 of L1
                actual_lm = l1;  // NOLINT
            }
            else if (l3 != 0 && problem_size <= 32768)
            {
                // we have both L2 and L3, and problem is small enough to be kept in L2
                // Let's choose m such that lhs's block fit in 1/3 of L2
                actual_lm = l2;  // NOLINT
                max_mc    = std::min<linalg_int>(576, max_mc);
            }
            linalg_int mc =
                std::min<linalg_int>(actual_lm / (3 * k * sizeof(value_t)), max_mc);  // NOLINT
            if (mc > static_cast<linalg_int>(Traits::mr))
            {
                mc -= mc % Traits::mr;
            }
            else if (mc == 0)
            {
                return l1;  // NOLINT
            }

            m = (m % mc) == 0
                    ? mc
                    : (mc - Traits::mr * ((mc /*-1*/ - (m % mc)) / (Traits::mr * (m / mc + 1))));
        }
    }  // namespace details

    return static_cast<linalg_int>(l1);
}

template <typename value_t,
    linalg_int Pack1,
    linalg_int Pack2,
    bool         ColMajor,
    bool         PanelMode = false>
void pack_lhs(  // NOLINT
    value_t*       block_A,
    const value_t* data,
    linalg_int   lda,
    linalg_int   depth,
    linalg_int   rows,
    linalg_int   stride = 0,
    linalg_int   offset = 0)
{
    using simd_t                             = typename simd<value_t>::simd_t;
    static constexpr linalg_int PacketSize = simd<value_t>::size;

    linalg_int count = 0;

    if constexpr (ColMajor)
    {
        const linalg_int peeled_mc3 =
            Pack1 >= 3 * PacketSize ? (rows / (3 * PacketSize)) * (3 * PacketSize) : 0;
        const linalg_int peeled_mc2 =
            Pack1 >= 2 * PacketSize
                ? peeled_mc3 + ((rows - peeled_mc3) / (2 * PacketSize)) * (2 * PacketSize)
                : 0;
        const linalg_int peeled_mc1 =
            Pack1 >= 1 * PacketSize ? (rows / (1 * PacketSize)) * (1 * PacketSize) : 0;
        const linalg_int peeled_mc0 = Pack2 >= 1 * PacketSize ? peeled_mc1
                                        : Pack2 > 1             ? (rows / Pack2) * Pack2
                                                                : 0;

        linalg_int i = 0;

        // Pack 3 packets
        if constexpr (Pack1 >= 3 * PacketSize)
        {
            for (; i < peeled_mc3; i += 3 * PacketSize)
            {
                if (PanelMode)
                {
                    count += (3 * PacketSize) * offset;
                }

                for (linalg_int k = 0; k < depth; k++)
                {
                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i + 0 * PacketSize, k, lda),
                        block_A + count);
                    count += PacketSize;

                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i + 1 * PacketSize, k, lda),
                        block_A + count);
                    count += PacketSize;

                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i + 2 * PacketSize, k, lda),
                        block_A + count);
                    count += PacketSize;
                }
                if (PanelMode)
                {
                    count += (3 * PacketSize) * (stride - offset - depth);
                }
            }
        }
        // Pack 2 packets
        if constexpr (Pack1 >= 2 * PacketSize)
        {
            for (; i < peeled_mc2; i += 2 * PacketSize)
            {
                if (PanelMode)
                {
                    count += (2 * PacketSize) * offset;
                }

                for (linalg_int k = 0; k < depth; k++)
                {
                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i + 0 * PacketSize, k, lda),
                        block_A + count);
                    count += PacketSize;

                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i + 1 * PacketSize, k, lda),
                        block_A + count);
                    count += PacketSize;
                }
                if (PanelMode)
                {
                    count += (2 * PacketSize) * (stride - offset - depth);
                }
            }
        }
        // Pack 1 packets
        if constexpr (Pack1 >= 1 * PacketSize)
        {
            for (; i < peeled_mc1; i += 1 * PacketSize)
            {
                if (PanelMode)
                {
                    count += (1 * PacketSize) * offset;
                }

                for (linalg_int k = 0; k < depth; k++)
                {
                    simd<value_t>::copy(
                        &element<value_t, ColMajor>(data, i, k, lda), block_A + count);
                    count += PacketSize;
                }
                if (PanelMode)
                {
                    count += (1 * PacketSize) * (stride - offset - depth);
                }
            }
        }
        // Pack scalars
        if constexpr (Pack2 < PacketSize && Pack2 > 1)
        {
            for (; i < peeled_mc0; i += Pack2)
            {
                if (PanelMode)
                {
                    count += Pack2 * offset;
                }

                for (linalg_int k = 0; k < depth; k++)
                {
                    for (linalg_int w = 0; w < Pack2; w++)
                    {
                        block_A[count++] = element<value_t, ColMajor>(data, i + w, k, lda);
                    }
                }

                if (PanelMode)
                {
                    count += Pack2 * (stride - offset - depth);
                }
            }
        }

        for (; i < rows; i++)  // NOLINT
        {
            if (PanelMode)
            {
                count += offset;
            }
            for (linalg_int k = 0; k < depth; k++)
            {
                block_A[count++] = element<value_t, ColMajor>(data, i, k, lda);
            }
            if (PanelMode)
            {
                count += (stride - offset - depth);
            }
        }
    }
    else
    {
        linalg_int pack = Pack1;
        linalg_int i    = 0;
        while (pack > 0)
        {
            const auto remaining_rows = rows - i;
            const auto peeled_mc      = i + (remaining_rows / pack) * pack;
            for (; i < peeled_mc; i += pack)
            {
                if (PanelMode)
                {
                    count += pack * offset;
                }

                const linalg_int peeled_k = (depth / PacketSize) * PacketSize;
                linalg_int       k        = 0;
                if (pack >= PacketSize)
                {
                    for (; k < peeled_k; k += PacketSize)
                    {
                        for (linalg_int m = 0; m < pack; m += PacketSize)
                        {
                            simd_t tmp[PacketSize];  // NOLINT
                            for (linalg_int p = 0; p < PacketSize; ++p)
                            {
                                simd<value_t>::loadu(
                                    &element<value_t, ColMajor>(data, i + p + m, k, lda), tmp[p]);
                            }

                            simd<value_t>::template ptranspose<PacketSize>(tmp);

                            for (linalg_int p = 0; p < PacketSize; ++p)
                            {
                                simd<value_t>::store(tmp[p], block_A + count + m + (pack)*p);
                            }
                        }
                        count += PacketSize * pack;
                    }
                }
                for (; k < depth; k++)
                {
                    linalg_int w = 0;
                    for (; w < pack - 3; w += 4)
                    {
                        block_A[count++] = element<value_t, ColMajor>(data, i + w, k, lda);
                        block_A[count++] = element<value_t, ColMajor>(data, i + w + 1, k, lda);
                        block_A[count++] = element<value_t, ColMajor>(data, i + w + 2, k, lda);
                        block_A[count++] = element<value_t, ColMajor>(data, i + w + 3, k, lda);
                    }
                    if (pack % 4)
                    {
                        for (; w < pack; ++w)
                        {
                            block_A[count++] = element<value_t, ColMajor>(data, i + w, k, lda);
                        }
                    }
                }

                if (PanelMode)
                {
                    count += pack * (stride - offset - depth);
                }
            }

            pack -= PacketSize;
            if (pack < Pack2 && (pack + PacketSize) != Pack2)
            {
                pack = Pack2;
            }
        }

        for (; i < rows; i++)
        {
            if (PanelMode)
            {
                count += offset;
            }
            for (linalg_int k = 0; k < depth; k++)
            {
                block_A[count++] = element<value_t, ColMajor>(data, i, k, lda);
            }
            if (PanelMode)
            {
                count += (stride - offset - depth);
            }
        }
    }
}

template <typename value_t, linalg_int nr, linalg_int PacketSize, bool ColMajor, bool PanelMode>
void pack_rhs(  // NOLINT
    value_t*       blockB,
    const value_t* data,
    linalg_int   ldb,
    linalg_int   depth,
    linalg_int   cols,
    linalg_int   stride = 0,
    linalg_int   offset = 0)
{
    using simd_t = typename simd<value_t>::simd_t;

    const linalg_int packet_cols4 = nr >= 4 ? (cols / 4) * 4 : 0;
    const linalg_int peeled_k     = (depth / PacketSize) * PacketSize;

    linalg_int count = 0;
    if constexpr (ColMajor)
    {
        if constexpr (nr >= 4)
        {
            for (linalg_int j2 = 0; j2 < packet_cols4; j2 += 4)
            {
                // skip what we have before
                if (PanelMode)
                {
                    count += 4 * offset;
                }

                const auto* const dm0 = &element<value_t, ColMajor>(data, 0, j2, ldb);
                const auto* const dm1 = &element<value_t, ColMajor>(data, 0, j2 + 1, ldb);
                const auto* const dm2 = &element<value_t, ColMajor>(data, 0, j2 + 2, ldb);
                const auto* const dm3 = &element<value_t, ColMajor>(data, 0, j2 + 3, ldb);
                linalg_int      k   = 0;
                if constexpr ((PacketSize % 4) == 0)
                {
                    for (; k < peeled_k; k += PacketSize)
                    {
                        simd_t tmp[PacketSize];  // NOLINT

                        simd<value_t>::loadu(dm0 + k, tmp[0]);
                        simd<value_t>::loadu(dm1 + k, tmp[1]);
                        simd<value_t>::loadu(dm2 + k, tmp[2]);
                        simd<value_t>::loadu(dm3 + k, tmp[3]);

                        constexpr linalg_int Np =
                            (PacketSize % 4) == 0 ? 4 : PacketSize;  // NOLINT

                        simd<value_t>::template ptranspose<Np>(tmp);
                        simd<value_t>::storeu(tmp[0], blockB + count);
                        simd<value_t>::storeu(tmp[1], blockB + count + PacketSize);
                        simd<value_t>::storeu(tmp[2], blockB + count + 2 * PacketSize);
                        simd<value_t>::storeu(tmp[3], blockB + count + 3 * PacketSize);
                        count += 4 * PacketSize;
                    }
                }
                for (; k < depth; k++)  // NOLINT
                {
                    blockB[count]     = dm0[k];
                    blockB[count + 1] = dm1[k];
                    blockB[count + 2] = dm2[k];
                    blockB[count + 3] = dm3[k];
                    count += 4;
                }
                // skip what we have after
                if (PanelMode)
                {
                    count += 4 * (stride - offset - depth);
                }
            }
        }

        // copy the remaining columns one at a time (nr==1)
        for (linalg_int j2 = packet_cols4; j2 < cols; ++j2)
        {
            if (PanelMode)
            {
                count += offset;
            }
            const auto* const dm0 = &element<value_t, ColMajor>(data, 0, j2, ldb);
            for (linalg_int k = 0; k < depth; k++)
            {
                blockB[count] = dm0[k];
                count += 1;
            }
            if (PanelMode)
            {
                count += (stride - offset - depth);
            }
        }
    }
    else
    {
        if constexpr (nr >= 4)
        {
            for (linalg_int j2 = 0; j2 < packet_cols4; j2 += 4)
            {
                // skip what we have before
                if (PanelMode)
                {
                    count += 4 * offset;
                }
                for (linalg_int k = 0; k < depth; k++)
                {
                    if constexpr (PacketSize == 4)
                    {
                        simd_t A;
                        simd<value_t>::loadu(&element<value_t, ColMajor>(data, k, j2, ldb), A);
                        simd<value_t>::storeu(A, blockB + count);
                        count += PacketSize;
                    }
                    else
                    {
                        const auto* const sub_matrix =
                            &element<value_t, ColMajor>(data, k, j2, ldb);
                        blockB[count + 0] = sub_matrix[0];
                        blockB[count + 1] = sub_matrix[1];
                        blockB[count + 2] = sub_matrix[2];
                        blockB[count + 3] = sub_matrix[3];
                        count += 4;
                    }
                }
                // skip what we have after
                if (PanelMode)
                {
                    count += 4 * (stride - offset - depth);
                }
            }
        }
        // copy the remaining columns one at a time (nr==1)
        for (linalg_int j2 = packet_cols4; j2 < cols; ++j2)
        {
            if (PanelMode)
            {
                count += offset;
            }
            for (linalg_int k = 0; k < depth; k++)
            {
                blockB[count] = element<value_t, ColMajor>(data, k, j2, ldb);
                count += 1;
            }
            if (PanelMode)
            {
                count += stride - offset - depth;
            }
        }
    }
}

template <typename value_t, linalg_int mr, linalg_int nr>
void gemm(  // NOLINT
    value_t*           res,
    const value_t*     blockA,
    const value_t*     blockB,
    linalg_int       ldc,
    linalg_int       rows,
    linalg_int       depth,
    linalg_int       cols,
    const linalg_int l1_cache,
    linalg_int       strideA = -1,
    linalg_int       strideB = -1,
    linalg_int       offsetA = 0,
    linalg_int       offsetB = 0)  // NOLINT
{
    using Traits = gemm_traits<value_t>;
    using simd_t = typename simd<value_t>::simd_t;
    if (strideA == -1)
    {
        strideA = depth;
    }
    if (strideB == -1)
    {
        strideB = depth;
    }

    static constexpr linalg_int prefetch_res_offset = 32 / sizeof(value_t);
    static constexpr linalg_int lhs_3offset         = 3 * Traits::LhsProgress;
    static constexpr linalg_int lhs_2offset         = 2 * Traits::LhsProgress;

    const linalg_int packet_cols4 = nr >= 4 ? (cols / 4) * 4 : 0;
    const linalg_int peeled_mc3 = mr >= lhs_3offset ? (rows / (lhs_3offset)) * (lhs_3offset) : 0;
    const linalg_int peeled_mc2 =
        mr >= lhs_2offset ? peeled_mc3 + ((rows - peeled_mc3) / (lhs_2offset)) * (lhs_2offset) : 0;
    const linalg_int peeled_mc1 =
        mr >= 1 * Traits::LhsProgress ? (rows / Traits::LhsProgress) * (Traits::LhsProgress) : 0;
    enum
    {
        pk = 8
    };  // NOTE Such a large peeling factor is important for large matrices (~ +5% when >1000 on
    // Haswell)
    const linalg_int peeled_kc = depth & ~(pk - 1);

    //---------- Process 3 * LhsProgress rows at once ----------
    // This corresponds to 3*LhsProgress x nr register blocks.
    // Usually, make sense only with FMA
    if constexpr (mr >= lhs_3offset)
    {
        const linalg_int actual_panel_rows = (lhs_3offset)*std::max<linalg_int>(1,
            ((l1_cache - sizeof(value_t) * mr * nr -
                 static_cast<long long>(depth) * nr * sizeof(value_t)) /
                (depth * sizeof(value_t) * lhs_3offset)));

        for (linalg_int i1 = 0; i1 < peeled_mc3; i1 += actual_panel_rows)
        {
            const linalg_int actual_panel_end =
                std::min<linalg_int>(i1 + actual_panel_rows, peeled_mc3);
            for (linalg_int j2 = 0; j2 < packet_cols4; j2 += nr)
            {
                for (linalg_int i = i1; i < actual_panel_end; i += lhs_3offset)
                {
                    // We selected a 3*Traits::LhsProgress x nr micro block of res which is entirely
                    // stored into 3 x nr registers.

                    const value_t* blA = &blockA[i * strideA + offsetA * (lhs_3offset)];
                    simd<value_t>::prefetch(&blA[0]);

                    // gets res block as register
                    simd_t C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11;  // NOLINT
                    simd<value_t>::set(static_cast<value_t>(0.), C0);
                    simd<value_t>::set(static_cast<value_t>(0.), C1);
                    simd<value_t>::set(static_cast<value_t>(0.), C2);
                    simd<value_t>::set(static_cast<value_t>(0.), C3);
                    simd<value_t>::set(static_cast<value_t>(0.), C4);
                    simd<value_t>::set(static_cast<value_t>(0.), C5);
                    simd<value_t>::set(static_cast<value_t>(0.), C6);
                    simd<value_t>::set(static_cast<value_t>(0.), C7);
                    simd<value_t>::set(static_cast<value_t>(0.), C8);
                    simd<value_t>::set(static_cast<value_t>(0.), C9);
                    simd<value_t>::set(static_cast<value_t>(0.), C10);
                    simd<value_t>::set(static_cast<value_t>(0.), C11);

                    auto* r0 = &element<value_t>(res, i, j2, ldc);
                    auto* r1 = &element<value_t>(res, i, j2 + 1, ldc);
                    auto* r2 = &element<value_t>(res, i, j2 + 2, ldc);
                    auto* r3 = &element<value_t>(res, i, j2 + 3, ldc);

                    simd<value_t>::prefetch(r0);
                    simd<value_t>::prefetch(r1);
                    simd<value_t>::prefetch(r2);
                    simd<value_t>::prefetch(r3);

                    // performs "inner" products
                    const value_t* blB = &blockB[j2 * strideB + offsetB * nr];
                    simd<value_t>::prefetch(&blB[0]);
                    simd_t A0;
                    simd_t A1;

                    for (linalg_int k = 0; k < peeled_kc; k += pk)
                    {
                        simd_t B_0;
                        simd_t A2;

#define EIGEN_GEBP_ONESTEP(K)                                                                      \
    do                                                                                             \
    {                                                                                              \
        simd<value_t>::prefetch(blA + (3 * (K) + 16) * Traits::LhsProgress);                       \
        simd<value_t>::load(&blA[(0 + 3 * (K)) * Traits::LhsProgress], A0);                        \
        simd<value_t>::load(&blA[(1 + 3 * (K)) * Traits::LhsProgress], A1);                        \
        simd<value_t>::load(&blA[(2 + 3 * (K)) * Traits::LhsProgress], A2);                        \
        simd<value_t>::set(*(blB + (0 + 4 * (K)) * Traits::RhsProgress), B_0);                     \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        simd<value_t>::fma(A1, B_0, C4, C4);                                                       \
        simd<value_t>::fma(A2, B_0, C8, C8);                                                       \
        simd<value_t>::set(*(blB + (1 + 4 * (K)) * Traits::RhsProgress), B_0);                     \
        simd<value_t>::fma(A0, B_0, C1, C1);                                                       \
        simd<value_t>::fma(A1, B_0, C5, C5);                                                       \
        simd<value_t>::fma(A2, B_0, C9, C9);                                                       \
        simd<value_t>::set(*(blB + (2 + 4 * (K)) * Traits::RhsProgress), B_0);                     \
        simd<value_t>::fma(A0, B_0, C2, C2);                                                       \
        simd<value_t>::fma(A1, B_0, C6, C6);                                                       \
        simd<value_t>::fma(A2, B_0, C10, C10);                                                     \
        simd<value_t>::set(*(blB + (3 + 4 * (K)) * Traits::RhsProgress), B_0);                     \
        simd<value_t>::fma(A0, B_0, C3, C3);                                                       \
        simd<value_t>::fma(A1, B_0, C7, C7);                                                       \
        simd<value_t>::fma(A2, B_0, C11, C11);                                                     \
    } while (false)

                        simd<value_t>::prefetch(blB);
                        EIGEN_GEBP_ONESTEP(0);
                        EIGEN_GEBP_ONESTEP(1);
                        EIGEN_GEBP_ONESTEP(2);
                        EIGEN_GEBP_ONESTEP(3);
                        EIGEN_GEBP_ONESTEP(4);
                        EIGEN_GEBP_ONESTEP(5);
                        EIGEN_GEBP_ONESTEP(6);
                        EIGEN_GEBP_ONESTEP(7);

                        blB += pk * 4 * Traits::RhsProgress;
                        blA += pk * lhs_3offset;
                    }
                    // process remaining peeled loop
                    for (linalg_int k = peeled_kc; k < depth; k++)
                    {
                        simd_t B_0;
                        simd_t A2;
                        EIGEN_GEBP_ONESTEP(0);
                        blB += 4 * Traits::RhsProgress;
                        blA += lhs_3offset;
                    }

#undef EIGEN_GEBP_ONESTEP
                    simd_t R0;
                    simd<value_t>::loadu(r0, R0);
                    simd<value_t>::add(R0, C0, R0);
                    simd<value_t>::storeu(R0, r0);

                    simd<value_t>::loadu(r0 + Traits::size, R0);
                    simd<value_t>::add(R0, C4, R0);
                    simd<value_t>::storeu(R0, r0 + Traits::size);

                    simd<value_t>::loadu(r0 + 2 * Traits::size, R0);
                    simd<value_t>::add(R0, C8, R0);
                    simd<value_t>::storeu(R0, r0 + 2 * Traits::size);

                    simd<value_t>::loadu(r1, R0);
                    simd<value_t>::add(R0, C1, R0);
                    simd<value_t>::storeu(R0, r1);

                    simd<value_t>::loadu(r1 + Traits::size, R0);
                    simd<value_t>::add(R0, C5, R0);
                    simd<value_t>::storeu(R0, r1 + Traits::size);

                    simd<value_t>::loadu(r1 + 2 * Traits::size, R0);
                    simd<value_t>::add(R0, C9, R0);
                    simd<value_t>::storeu(R0, r1 + 2 * Traits::size);

                    simd<value_t>::loadu(r2, R0);
                    simd<value_t>::add(R0, C2, R0);
                    simd<value_t>::storeu(R0, r2);

                    simd<value_t>::loadu(r2 + Traits::size, R0);
                    simd<value_t>::add(R0, C6, R0);
                    simd<value_t>::storeu(R0, r2 + Traits::size);

                    simd<value_t>::loadu(r2 + 2 * Traits::size, R0);
                    simd<value_t>::add(R0, C10, R0);
                    simd<value_t>::storeu(R0, r2 + 2 * Traits::size);

                    simd<value_t>::loadu(r3, R0);
                    simd<value_t>::add(R0, C3, R0);
                    simd<value_t>::storeu(R0, r3);

                    simd<value_t>::loadu(r3 + Traits::size, R0);
                    simd<value_t>::add(R0, C7, R0);
                    simd<value_t>::storeu(R0, r3 + Traits::size);

                    simd<value_t>::loadu(r3 + 2 * Traits::size, R0);
                    simd<value_t>::add(R0, C11, R0);
                    simd<value_t>::storeu(R0, r3 + 2 * Traits::size);
                }
            }

            // Deal with remaining columns of the rhs
            for (linalg_int j2 = packet_cols4; j2 < cols; j2++)
            {
                for (linalg_int i = i1; i < actual_panel_end; i += lhs_3offset)
                {
                    // One column at a time
                    const value_t* blA = &blockA[i * strideA + offsetA * (lhs_3offset)];
                    simd<value_t>::prefetch(&blA[0]);

                    // gets res block as register
                    simd_t C0, C4, C8;  // NOLINT
                    simd<value_t>::set(static_cast<value_t>(0.), C0);
                    simd<value_t>::set(static_cast<value_t>(0.), C4);
                    simd<value_t>::set(static_cast<value_t>(0.), C8);

                    auto* r0 = &element<value_t>(res, i, j2, ldc);
                    simd<value_t>::prefetch(r0);

                    // performs "inner" products
                    const value_t* blB = &blockB[j2 * strideB + offsetB];
                    simd_t         A0, A1, A2;  // NOLINT

                    for (linalg_int k = 0; k < peeled_kc; k += pk)
                    {
                        simd_t B_0;
#define LINALG_GEMM_ONESTEP(K)                                                                     \
    do                                                                                             \
    {                                                                                              \
        simd<value_t>::load(&blA[(0 + 3 * (K)) * Traits::LhsProgress], A0);                        \
        simd<value_t>::load(&blA[(1 + 3 * (K)) * Traits::LhsProgress], A1);                        \
        simd<value_t>::load(&blA[(2 + 3 * (K)) * Traits::LhsProgress], A2);                        \
        simd<value_t>::set(blB[(0 + (K)) * Traits::RhsProgress], B_0);                             \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        simd<value_t>::fma(A1, B_0, C4, C4);                                                       \
        simd<value_t>::fma(A2, B_0, C8, C8);                                                       \
    } while (false)

                        LINALG_GEMM_ONESTEP(0);
                        LINALG_GEMM_ONESTEP(1);
                        LINALG_GEMM_ONESTEP(2);
                        LINALG_GEMM_ONESTEP(3);
                        LINALG_GEMM_ONESTEP(4);
                        LINALG_GEMM_ONESTEP(5);
                        LINALG_GEMM_ONESTEP(6);
                        LINALG_GEMM_ONESTEP(7);

                        blB += pk * Traits::RhsProgress;
                        blA += pk * lhs_3offset;
                    }

                    // process remaining peeled loop
                    for (linalg_int k = peeled_kc; k < depth; k++)
                    {
                        simd_t B_0;
                        LINALG_GEMM_ONESTEP(0);
                        blB += Traits::RhsProgress;
                        blA += lhs_3offset;
                    }
#undef LINALG_GEMM_ONESTEP
                    simd_t R0;
                    simd<value_t>::loadu(r0, R0);
                    simd<value_t>::add(R0, C0, R0);
                    simd<value_t>::storeu(R0, r0);

                    simd<value_t>::loadu(r0 + Traits::size, R0);
                    simd<value_t>::add(R0, C4, R0);
                    simd<value_t>::storeu(R0, r0 + Traits::size);

                    simd<value_t>::loadu(r0 + 2 * Traits::size, R0);
                    simd<value_t>::add(R0, C8, R0);
                    simd<value_t>::storeu(R0, r0 + 2 * Traits::size);
                }
            }
        }
    }

    //---------- Process 2 * LhsProgress rows at once ----------
    if constexpr (mr >= lhs_2offset)
    {
        const auto actual_panel_rows = (lhs_2offset)*std::max<linalg_int>(1,
            ((l1_cache - sizeof(value_t) * mr * nr -
                 static_cast<long long>(depth) * nr * sizeof(value_t)) /
                (depth * sizeof(value_t) * lhs_2offset)));

        for (linalg_int i1 = peeled_mc3; i1 < peeled_mc2; i1 += actual_panel_rows)
        {
            const auto actual_panel_end =
                std::min<linalg_int>(i1 + actual_panel_rows, peeled_mc2);
            for (linalg_int j2 = 0; j2 < packet_cols4; j2 += nr)
            {
                for (linalg_int i = i1; i < actual_panel_end; i += lhs_2offset)
                {
                    // We selected a 2*Traits::LhsProgress x nr micro block of res which is entirely
                    // stored into 2 x nr registers.

                    const value_t* blA = &blockA[i * strideA + offsetA * lhs_2offset];
                    simd<value_t>::prefetch(&blA[0]);

                    // gets res block as register
                    simd_t C0, C1, C2, C3, C4, C5, C6, C7;  // NOLINT
                    simd<value_t>::set(static_cast<value_t>(0.), C0);
                    simd<value_t>::set(static_cast<value_t>(0.), C1);
                    simd<value_t>::set(static_cast<value_t>(0.), C2);
                    simd<value_t>::set(static_cast<value_t>(0.), C3);
                    simd<value_t>::set(static_cast<value_t>(0.), C4);
                    simd<value_t>::set(static_cast<value_t>(0.), C5);
                    simd<value_t>::set(static_cast<value_t>(0.), C6);
                    simd<value_t>::set(static_cast<value_t>(0.), C7);

                    value_t* r0 = &element<value_t>(res, i, j2, ldc);
                    value_t* r1 = &element<value_t>(res, i, j2 + 1, ldc);
                    value_t* r2 = &element<value_t>(res, i, j2 + 2, ldc);
                    value_t* r3 = &element<value_t>(res, i, j2 + 3, ldc);

                    simd<value_t>::prefetch(r0 + prefetch_res_offset);
                    simd<value_t>::prefetch(r1 + prefetch_res_offset);
                    simd<value_t>::prefetch(r2 + prefetch_res_offset);
                    simd<value_t>::prefetch(r3 + prefetch_res_offset);

                    // performs "inner" products
                    const value_t* blB = &blockB[j2 * strideB + offsetB * nr];
                    simd<value_t>::prefetch(&blB[0]);
                    simd_t A0;
                    simd_t A1;

                    for (linalg_int k = 0; k < peeled_kc; k += pk)
                    {
                        simd_t B_0;
                        simd_t B1;
                        simd_t B2;
                        simd_t B3;

#define LINALG_GEMM_ONESTEP(K)                                                                     \
    do                                                                                             \
    {                                                                                              \
        simd<value_t>::load(&blA[(0 + 2 * (K)) * Traits::LhsProgress], A0);                        \
        simd<value_t>::load(&blA[(1 + 2 * (K)) * Traits::LhsProgress], A1);                        \
        const auto b_ptr = &blB[(0 + 4 * (K)) * Traits::RhsProgress];                              \
        simd<value_t>::broadcast(b_ptr, B_0, B1, B2, B3);                                          \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        simd<value_t>::fma(A1, B_0, C4, C4);                                                       \
        simd<value_t>::fma(A0, B1, C1, C1);                                                        \
        simd<value_t>::fma(A1, B1, C5, C5);                                                        \
        simd<value_t>::fma(A0, B2, C2, C2);                                                        \
        simd<value_t>::fma(A1, B2, C6, C6);                                                        \
        simd<value_t>::fma(A0, B3, C3, C3);                                                        \
        simd<value_t>::fma(A1, B3, C7, C7);                                                        \
    } while (false)

                        simd<value_t>::prefetch(blB + (48 + 0));
                        LINALG_GEMM_ONESTEP(0);
                        LINALG_GEMM_ONESTEP(1);
                        LINALG_GEMM_ONESTEP(2);
                        LINALG_GEMM_ONESTEP(3);
                        simd<value_t>::prefetch(blB + (48 + 16));
                        LINALG_GEMM_ONESTEP(4);
                        LINALG_GEMM_ONESTEP(5);
                        LINALG_GEMM_ONESTEP(6);
                        LINALG_GEMM_ONESTEP(7);

                        blB += pk * 4 * Traits::RhsProgress;
                        blA += pk * (lhs_2offset);
                    }
                    // process remaining peeled loop
                    for (linalg_int k = peeled_kc; k < depth; k++)
                    {
                        simd_t B_0;
                        simd_t B1;
                        simd_t B2;
                        simd_t B3;
                        LINALG_GEMM_ONESTEP(0);
                        blB += 4 * Traits::RhsProgress;
                        blA += lhs_2offset;
                    }
#undef LINALG_GEMM_ONESTEP

                    simd_t R0;
                    simd<value_t>::loadu(r0, R0);
                    simd<value_t>::add(R0, C0, R0);
                    simd<value_t>::storeu(R0, r0);

                    simd<value_t>::loadu(r0 + Traits::size, R0);
                    simd<value_t>::add(R0, C4, R0);
                    simd<value_t>::storeu(R0, r0 + Traits::size);

                    simd<value_t>::loadu(r1, R0);
                    simd<value_t>::add(R0, C1, R0);
                    simd<value_t>::storeu(R0, r1);

                    simd<value_t>::loadu(r1 + Traits::size, R0);
                    simd<value_t>::add(R0, C5, R0);
                    simd<value_t>::storeu(R0, r1 + Traits::size);

                    simd<value_t>::loadu(r2, R0);
                    simd<value_t>::add(R0, C2, R0);
                    simd<value_t>::storeu(R0, r2);

                    simd<value_t>::loadu(r2 + Traits::size, R0);
                    simd<value_t>::add(R0, C6, R0);
                    simd<value_t>::storeu(R0, r2 + Traits::size);

                    simd<value_t>::loadu(r3, R0);
                    simd<value_t>::add(R0, C3, R0);
                    simd<value_t>::storeu(R0, r3);

                    simd<value_t>::loadu(r3 + Traits::size, R0);
                    simd<value_t>::add(R0, C7, R0);
                    simd<value_t>::storeu(R0, r3 + Traits::size);
                }
            }

            // Deal with remaining columns of the rhs
            for (linalg_int j2 = packet_cols4; j2 < cols; j2++)
            {
                for (linalg_int i = i1; i < actual_panel_end; i += lhs_2offset)
                {
                    // One column at a time
                    const value_t* blA = &blockA[i * strideA + offsetA * (lhs_2offset)];
                    simd<value_t>::prefetch(&blA[0]);

                    // gets res block as register
                    simd_t C0;
                    simd_t C4;
                    simd<value_t>::set(static_cast<value_t>(0.), C0);
                    simd<value_t>::set(static_cast<value_t>(0.), C4);

                    auto* r0 = &element<value_t>(res, i, j2, ldc);
                    simd<value_t>::prefetch(r0 + prefetch_res_offset);

                    // performs "inner" products
                    const value_t* blB = &blockB[j2 * strideB + offsetB];
                    simd_t         A0;
                    simd_t         A1;

                    for (linalg_int k = 0; k < peeled_kc; k += pk)
                    {
                        LINALG_ASM_COMMENT("begin gebp micro kernel 2pX1");
                        simd_t B_0;

#define LINALG_GEMM_ONESTEP(K)                                                                     \
    do                                                                                             \
    {                                                                                              \
        simd<value_t>::load(&blA[(0 + 2 * (K)) * Traits::LhsProgress], A0);                        \
        simd<value_t>::load(&blA[(1 + 2 * (K)) * Traits::LhsProgress], A1);                        \
        simd<value_t>::set(blB[(0 + (K)) * Traits::RhsProgress], B_0);                             \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        simd<value_t>::fma(A1, B_0, C4, C4);                                                       \
    } while (false)

                        LINALG_GEMM_ONESTEP(0);
                        LINALG_GEMM_ONESTEP(1);
                        LINALG_GEMM_ONESTEP(2);
                        LINALG_GEMM_ONESTEP(3);
                        LINALG_GEMM_ONESTEP(4);
                        LINALG_GEMM_ONESTEP(5);
                        LINALG_GEMM_ONESTEP(6);
                        LINALG_GEMM_ONESTEP(7);

                        blB += pk * Traits::RhsProgress;
                        blA += pk * lhs_2offset;
                    }

                    // process remaining peeled loop
                    for (linalg_int k = peeled_kc; k < depth; k++)
                    {
                        simd_t B_0;
                        LINALG_GEMM_ONESTEP(0);
                        blB += Traits::RhsProgress;
                        blA += lhs_2offset;
                    }
#undef LINALG_GEMM_ONESTEP

                    simd_t R0;
                    simd<value_t>::loadu(r0, R0);
                    simd<value_t>::add(R0, C0, R0);
                    simd<value_t>::storeu(R0, r0);

                    simd<value_t>::loadu(r0 + Traits::size, R0);
                    simd<value_t>::add(R0, C4, R0);
                    simd<value_t>::storeu(R0, r0 + Traits::size);
                }
            }
        }
    }
    //---------- Process 1 * LhsProgress rows at once ----------
    if constexpr (mr >= 1 * Traits::LhsProgress)
    {
        // loops on each largest micro horizontal panel of lhs (1*LhsProgress x depth)
        for (linalg_int i = peeled_mc2; i < peeled_mc1; i += Traits::LhsProgress)
        {
            // loops on each largest micro vertical panel of rhs (depth * nr)
            for (linalg_int j2 = 0; j2 < packet_cols4; j2 += nr)
            {
                // We select a 1*Traits::LhsProgress x nr micro block of res which is entirely
                // stored into 1 x nr registers.

                const value_t* blA = &blockA[i * strideA + offsetA * (1 * Traits::LhsProgress)];
                simd<value_t>::prefetch(&blA[0]);

                // gets res block as register
                simd_t C0;
                simd_t C1;
                simd_t C2;
                simd_t C3;
                simd<value_t>::set(static_cast<value_t>(0.), C0);
                simd<value_t>::set(static_cast<value_t>(0.), C1);
                simd<value_t>::set(static_cast<value_t>(0.), C2);
                simd<value_t>::set(static_cast<value_t>(0.), C3);

                auto* r0 = &element<value_t>(res, i, j2, ldc);
                auto* r1 = &element<value_t>(res, i, j2 + 1, ldc);
                auto* r2 = &element<value_t>(res, i, j2 + 2, ldc);
                auto* r3 = &element<value_t>(res, i, j2 + 3, ldc);

                simd<value_t>::prefetch(r0 + prefetch_res_offset);
                simd<value_t>::prefetch(r1 + prefetch_res_offset);
                simd<value_t>::prefetch(r2 + prefetch_res_offset);
                simd<value_t>::prefetch(r3 + prefetch_res_offset);

                // performs "inner" products
                const value_t* blB = &blockB[j2 * strideB + offsetB * nr];
                simd<value_t>::prefetch(&blB[0]);
                simd_t A0;

                for (linalg_int k = 0; k < peeled_kc; k += pk)
                {
                    LINALG_ASM_COMMENT("begin gebp micro kernel 1pX4");
                    simd_t B_0;
                    simd_t B1;
                    simd_t B2;
                    simd_t B3;

#define LINALG_GEMM_ONESTEP(K)                                                                     \
    do                                                                                             \
    {                                                                                              \
        LINALG_ASM_COMMENT("begin step of gebp micro kernel 1pX4");                                \
        LINALG_ASM_COMMENT("Note: these asm comments work around bug 935!");                       \
        simd<value_t>::load(&blA[(0 + 1 * (K)) * Traits::LhsProgress], A0);                        \
        const auto b_ptr = &blB[(0 + 4 * (K)) * Traits::RhsProgress];                              \
        simd<value_t>::broadcast(b_ptr, B_0, B1, B2, B3);                                          \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        simd<value_t>::fma(A0, B1, C1, C1);                                                        \
        simd<value_t>::fma(A0, B2, C2, C2);                                                        \
        simd<value_t>::fma(A0, B3, C3, C3);                                                        \
        LINALG_ASM_COMMENT("end step of gebp micro kernel 1pX4");                                  \
    } while (false)

                    simd<value_t>::prefetch(blB + (48 + 0));
                    LINALG_GEMM_ONESTEP(0);
                    LINALG_GEMM_ONESTEP(1);
                    LINALG_GEMM_ONESTEP(2);
                    LINALG_GEMM_ONESTEP(3);

                    simd<value_t>::prefetch(blB + (48 + 16));
                    LINALG_GEMM_ONESTEP(4);
                    LINALG_GEMM_ONESTEP(5);
                    LINALG_GEMM_ONESTEP(6);
                    LINALG_GEMM_ONESTEP(7);

                    blB += pk * 4 * Traits::RhsProgress;
                    blA += pk * 1 * Traits::LhsProgress;

                    LINALG_ASM_COMMENT("end gebp micro kernel 1pX4");
                }
                // process remaining peeled loop
                for (linalg_int k = peeled_kc; k < depth; k++)
                {
                    simd_t B_0;
                    simd_t B1;
                    simd_t B2;
                    simd_t B3;
                    LINALG_GEMM_ONESTEP(0);
                    blB += 4 * Traits::RhsProgress;
                    blA += 1 * Traits::LhsProgress;
                }
#undef LINALG_GEMM_ONESTEP

                simd_t R0;
                simd<value_t>::loadu(r0, R0);
                simd<value_t>::add(R0, C0, R0);
                simd<value_t>::storeu(R0, r0);

                simd<value_t>::loadu(r1, R0);
                simd<value_t>::add(R0, C1, R0);
                simd<value_t>::storeu(R0, r1);

                simd<value_t>::loadu(r2, R0);
                simd<value_t>::add(R0, C2, R0);
                simd<value_t>::storeu(R0, r2);

                simd<value_t>::loadu(r3, R0);
                simd<value_t>::add(R0, C3, R0);
                simd<value_t>::storeu(R0, r3);
            }

            // Deal with remaining columns of the rhs
            for (linalg_int j2 = packet_cols4; j2 < cols; j2++)
            {
                // One column at a time
                const value_t* blA = &blockA[i * strideA + offsetA * (1 * Traits::LhsProgress)];
                simd<value_t>::prefetch(&blA[0]);

                // gets res block as register
                simd_t C0;
                simd<value_t>::set(static_cast<value_t>(0.), C0);

                auto* r0 = &element<value_t>(res, i, j2, ldc);
                // performs "inner" products
                const value_t* blB = &blockB[j2 * strideB + offsetB];
                simd_t         A0;

                for (linalg_int k = 0; k < peeled_kc; k += pk)
                {
                    LINALG_ASM_COMMENT("begin gebp micro kernel 1pX1");
                    simd_t B_0;

#define LINALG_GEMM_ONESTEP(K)                                                                     \
    do                                                                                             \
    {                                                                                              \
        LINALG_ASM_COMMENT("begin step of gebp micro kernel 1pX1");                                \
        LINALG_ASM_COMMENT("Note: these asm comments work around bug 935!");                       \
        simd<value_t>::load(&blA[(0 + (K)) * Traits::LhsProgress], A0);                            \
        simd<value_t>::set(blB[(0 + (K)) * Traits::RhsProgress], B_0);                             \
        simd<value_t>::fma(A0, B_0, C0, C0);                                                       \
        LINALG_ASM_COMMENT("end step of gebp micro kernel 1pX1");                                  \
    } while (false);

                    LINALG_GEMM_ONESTEP(0);
                    LINALG_GEMM_ONESTEP(1);
                    LINALG_GEMM_ONESTEP(2);
                    LINALG_GEMM_ONESTEP(3);
                    LINALG_GEMM_ONESTEP(4);
                    LINALG_GEMM_ONESTEP(5);
                    LINALG_GEMM_ONESTEP(6);
                    LINALG_GEMM_ONESTEP(7);

                    blB += pk * Traits::RhsProgress;
                    blA += pk * 1 * Traits::LhsProgress;

                    LINALG_ASM_COMMENT("end gebp micro kernel 1pX1");
                }

                // process remaining peeled loop
                for (linalg_int k = peeled_kc; k < depth; k++)
                {
                    simd_t B_0;
                    LINALG_GEMM_ONESTEP(0);
                    blB += Traits::RhsProgress;
                    blA += Traits::LhsProgress;
                }
#undef LINALG_GEMM_ONESTEP
                simd_t R0;
                simd<value_t>::loadu(r0, R0);
                simd<value_t>::add(R0, C0, R0);
                simd<value_t>::storeu(R0, r0);
            }
        }
    }
    //---------- Process remaining rows, 1 at once ----------
    if (peeled_mc1 < rows)
    {
        // loop on each panel of the rhs
        for (linalg_int j2 = 0; j2 < packet_cols4; j2 += nr)
        {
            // loop on each row of the lhs (1*LhsProgress x depth)
            for (linalg_int i = peeled_mc1; i < rows; i += 1)
            {
                const value_t* blA = &blockA[i * strideA + offsetA];
                simd<value_t>::prefetch(&blA[0]);
                const value_t* blB = &blockB[j2 * strideB + offsetB * nr];

                // The following piece of code wont work for 512 bit registers
                // Moreover, if LhsProgress==8 it assumes that there is a half tmp of the
                // same size as nr (which is currently 4) for the return type.
                static constexpr linalg_int packet_half_size = simd<value_t>::half_size;
                if constexpr ((Traits::LhsProgress % 4) == 0 && (Traits::LhsProgress <= 8) &&
                              (Traits::LhsProgress != 8 || packet_half_size == nr))
                {
                    simd_t C0;
                    simd_t C1;
                    simd_t C2;
                    simd_t C3;
                    simd<value_t>::set(static_cast<value_t>(0.), C0);
                    simd<value_t>::set(static_cast<value_t>(0.), C1);
                    simd<value_t>::set(static_cast<value_t>(0.), C2);
                    simd<value_t>::set(static_cast<value_t>(0.), C3);

                    const linalg_int spk   = std::max<linalg_int>(1, Traits::LhsProgress / 4);
                    const linalg_int endk  = (depth / spk) * spk;
                    const linalg_int endk4 = (depth / (spk * 4)) * (spk * 4);

                    linalg_int k = 0;
                    for (; k < endk4; k += 4 * spk)
                    {
                        simd_t A0;
                        simd_t A1;
                        simd_t B_0;
                        simd_t B_1;

                        simd<value_t>::loadu(blB, A0);
                        simd<value_t>::loadu(blB + Traits::LhsProgress, A1);
                        simd<value_t>::ploadquad(blA + 0 * spk, B_0);
                        simd<value_t>::ploadquad(blA + 1 * spk, B_1);
                        simd<value_t>::fma(A0, B_0, C0, C0);
                        simd<value_t>::fma(A1, B_1, C1, C1);

                        simd<value_t>::loadu(blB + lhs_2offset, A0);
                        simd<value_t>::loadu(blB + lhs_3offset, A1);
                        simd<value_t>::ploadquad(blA + 2 * spk, B_0);
                        simd<value_t>::ploadquad(blA + 3 * spk, B_1);
                        simd<value_t>::fma(A0, B_0, C2, C2);
                        simd<value_t>::fma(A1, B_1, C3, C3);

                        blB += 4 * Traits::LhsProgress;
                        blA += 4 * spk;
                    }
                    simd<value_t>::add(C0, C1, C0);
                    simd<value_t>::add(C2, C3, C2);
                    simd<value_t>::add(C0, C2, C0);
                    /* C0 = simd<value_t>::add(
                         simd<value_t>::add(C0, C1), simd<value_t>::add(C2,
                       C3));*/

                    for (; k < endk; k += spk)
                    {
                        simd_t A0;
                        simd_t B_0;

                        simd<value_t>::loadu(blB, A0);
                        simd<value_t>::ploadquad(blA, B_0);
                        simd<value_t>::fma(A0, B_0, C0, C0);

                        blB += Traits::LhsProgress;
                        blA += spk;
                    }

                    auto* to = &element<value_t>(res, i, j2, ldc);

                    if constexpr (Traits::LhsProgress == 8)
                    {
                        using simd_half_t = typename simd<value_t>::simd_half_t;
                        simd_half_t R;
                        simd<value_t>::gather(to, ldc, R);
                        if (depth - endk > 0)
                        {
                            // We have to handle the last row of the rhs which corresponds to a
                            // half-tmp
                            const auto a = simd<value_t>::loadu_half(blB);
                            const auto b = simd<value_t>::set(blA);
                            auto       c = simd<value_t>::predux_downto4(C0);
                            simd<value_t>::fma(a, b, c);
                            simd<value_t>::add(R, c, R);
                        }
                        else
                        {
                            auto c = simd<value_t>::predux_downto4(C0);
                            simd<value_t>::add(R, c, R);
                        }
                        simd<value_t>::scatter(R, ldc, to);
                    }
                    else
                    {
                        simd_t R;
                        simd<value_t>::gather(to, ldc, R);
                        simd<value_t>::add(R, C0, R);
                        simd<value_t>::scatter(R, ldc, to);
                    }
                }
                else  // scalar path
                {
                    // get a 1 x 4 res block as registers
                    value_t C0 = 0.;
                    value_t C1 = 0.;
                    value_t C2 = 0.;
                    value_t C3 = 0.;

                    for (linalg_int k = 0; k < depth; k++)
                    {
                        const auto& A0 = blA[k];
                        {
                            const auto& B_0 = blB[0];
                            const auto& B_1 = blB[1];

                            C0 += A0 * B_0;
                            C1 += A0 * B_1;
                        }
                        {
                            const auto& B_0 = blB[2];
                            const auto& B_1 = blB[3];
                            C2 += A0 * B_0;
                            C3 += A0 * B_1;
                        }
                        blB += 4;
                    }
                    element<value_t>(res, i, j2, ldc) += C0;
                    element<value_t>(res, i, j2 + 1, ldc) += C1;
                    element<value_t>(res, i, j2 + 2, ldc) += C2;
                    element<value_t>(res, i, j2 + 3, ldc) += C3;
                }
            }
        }
        // remaining columns
        for (linalg_int j2 = packet_cols4; j2 < cols; j2++)
        {
            // loop on each row of the lhs (1*LhsProgress x depth)
            for (linalg_int i = peeled_mc1; i < rows; i++)
            {
                const value_t* blA = &blockA[i * strideA + offsetA];
                simd<value_t>::prefetch(&blA[0]);
                // gets a 1 x 1 res block as registers
                value_t        C0  = 0.;
                const value_t* blB = &blockB[j2 * strideB + offsetB];
                for (linalg_int k = 0; k < depth; k++)
                {
                    const auto& A0  = blA[k];
                    const auto& B_0 = blB[k];
                    C0 += A0 * B_0;
                }
                element<value_t>(res, i, j2, ldc) += C0;
            }
        }
    }
}
#endif

template <typename value_t, bool transpose_a, bool transpose_b>
void matrix_multiplication_seq(  // NOLINT
    linalg_int   rows,
    linalg_int   columns,
    linalg_int   depth,
    value_t const* a,
    linalg_int   lda,
    value_t const* b,
    linalg_int   ldb,
    value_t*       c,
    linalg_int   ldc)
{
#if defined(LINALG_VECTORIZED)
    if (rows + columns + depth > 20)
    {
        ldc = rows;

        auto       kc       = depth;
        auto       mc       = rows;
        auto       nc       = columns;
        const auto l1_cache = blocking_sizes<value_t>(kc, mc, nc, 1);

        mc = std::min<linalg_int>(rows, mc);     // cache block size along the M direction
        nc = std::min<linalg_int>(columns, nc);  // cache block size along the N direction

        linalg_int sizeA = mc * kc;  // NOLINT
        linalg_int sizeB = kc * nc;  // NOLINT

        using allocator_t = allocator<value_t>;

        value_t* block_A = allocator_t::allocate(sizeA);
        value_t* block_B = allocator_t::allocate(sizeB);
        value_t* result  = rows == columns ? c : allocator_t::allocate(rows * columns);

        std::fill_n(result, rows * columns, 0.);

        const bool pack_rhs_once = mc != rows && kc == depth && nc == columns;

        // For each horizontal panel of the rhs, and corresponding panel of the lhs...
        for (linalg_int i = 0; i < rows; i += mc)
        {
            const linalg_int actual_mc = std::min<linalg_int>(i + mc, rows) - i;

            for (linalg_int k = 0; k < depth; k += kc)
            {
                const linalg_int actual_kc = std::min<linalg_int>(k + kc, depth) - k;

                // OK, here we have selected one horizontal panel of rhs and one vertical panel
                // of lhs.
                // => Pack lhs's panel into a sequential chunk of memory (L2/L3 caching)
                // Note that this panel will be read as many times as the number of blocks in
                // the rhs's horizontal panel which is, in practice, a very low number.
                pack_lhs<value_t,
                    gemm_traits<value_t>::mr,
                    gemm_traits<value_t>::LhsProgress,
                    transpose_a>(block_A,
                    &element<value_t, transpose_a>(a, i, k, lda),
                    lda,
                    actual_kc,
                    actual_mc);

                // For each kc x nc block of the rhs's horizontal panel...
                for (linalg_int j = 0; j < columns; j += nc)
                {
                    const linalg_int actual_nc = std::min<linalg_int>(j + nc, columns) - j;

                    // We pack the rhs's block into a sequential chunk of memory (L2 caching)
                    // Note that this block will be read a very high number of times, which is
                    // equal to the number of micro horizontal panel of the large rhs's panel
                    // (e.g., rows/12 times).
                    if ((!pack_rhs_once) || i == 0)
                    {
                        pack_rhs<value_t,
                            gemm_traits<value_t>::nr,
                            simd<value_t>::size,
                            transpose_b,
                            false>(block_B,
                            &element<value_t, transpose_b>(b, k, j, ldb),
                            ldb,
                            actual_kc,
                            actual_nc);
                    }

                    // Everything is packed, we can now call the panel * block kernel:
                    gemm<value_t, gemm_traits<value_t>::mr, gemm_traits<value_t>::nr>(
                        &element<value_t>(result, i, j, ldc),
                        block_A,
                        block_B,
                        ldc,
                        actual_mc,
                        actual_kc,
                        actual_nc,
                        l1_cache);
                }
            }
        }
        if (columns != rows)
        {
            for (linalg_int i = 0; i < rows; i++)
            {
                for (linalg_int j = 0; j < columns; j++)
                {
                    element<value_t, false>(c, i, j, columns) = element<value_t>(result, i, j, ldc);
                }
            }

            allocator_t::free(result);
        }
        else
        {
            auto n = rows;
            for (linalg_int i = 0; i < n; i++)
            {
                for (linalg_int j = i; j < n; j++)
                {
                    auto& alpha = element<value_t>(c, i, j, n);
                    auto& beta  = element<value_t>(c, j, i, n);
                    std::swap(alpha, beta);
                }
            }
        }

        allocator_t::free(block_B);
        allocator_t::free(block_A);
    }
    else
#endif
    {
        using size_type = linalg_int;
        if constexpr (!transpose_a && !transpose_b)
        {
            for (size_type i = 0; i < rows; ++i)
            {
                auto* c_i = &C(i, 0);
                auto* a_i = &A(i, 0);
                std::fill_n(c_i, columns, static_cast<value_t>(0.));

                for (size_type k = 0; k < depth; ++k)
                {
                    auto*      b_k      = &B(k, 0);
                    const auto alpha_ik = a_i[k];
                    for (size_type j = 0; j < columns; ++j)
                    {
                        c_i[j] += alpha_ik * b_k[j];
                    }
                }
            }
            return;
        }
        else if constexpr (transpose_a && !transpose_b)
        {
            std::fill_n(&C(0, 0), columns * rows, static_cast<value_t>(0.));
            for (size_type k = 0; k < depth; ++k)
            {
                auto* b_k = &B(k, 0);
                auto* a_k = &A(k, 0);
                for (size_type i = 0; i < rows; ++i)
                {
                    const auto alpha_ki = a_k[i];
                    auto*      c_i      = &C(i, 0);
                    for (size_type j = 0; j < columns; ++j)
                    {
                        c_i[j] += alpha_ki * b_k[j];
                    }
                }
            }
            return;
        }
        else if constexpr (!transpose_a && transpose_b)
        {
            for (size_type i = 0; i < rows; ++i)
            {
                for (size_type j = 0; j < columns; ++j)
                {
                    auto& c0                = C(i, j);
                    c0                      = 0;
                    const auto* const alpha = &A(i, 0);
                    const auto* const beta  = &B(j, 0);
                    for (size_type k = 0; k < depth; ++k)
                    {
                        c0 += alpha[k] * beta[k];
                    }
                }
            }
            return;
        }
        else
        {
            for (size_type i = 0; i < rows; ++i)
            {
                for (size_type j = 0; j < columns; ++j)
                {
                    auto& c0 = C(i, j);
                    c0       = 0;
                    for (size_type k = 0; k < depth; ++k)
                    {
                        c0 += A(k, i) * B(j, k);
                    }
                }
            }
            return;
        }
    }
}
}  // namespace
// C(i,j) =C[i*ldc + j] j belong to [0,columns] and i to [0,rows]
void matmul_scalar_f32(bool transpose_a,
    bool                    transpose_b,
    linalg_int            rows,
    linalg_int            columns,
    linalg_int            depth,
    float const*            a,
    linalg_int            lda,
    float const*            b,
    linalg_int            ldb,
    float*                  c,
    linalg_int            ldc)
{
#if defined(LINALG_VECTORIZED)

    if (transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, true, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, true, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, false, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, false, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
#else
    if (transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, true, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, true, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && transpose_b)
    {
        matrix_multiplication_seq<float, false, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<float, false, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
#endif
}

void matmul_scalar_f64(bool transpose_a,
    bool                    transpose_b,
    linalg_int            rows,
    linalg_int            columns,
    linalg_int            depth,
    double const*           a,
    linalg_int            lda,
    double const*           b,
    linalg_int            ldb,
    double*                 c,
    linalg_int            ldc)
{
    if (transpose_a && transpose_b)
    {
        matrix_multiplication_seq<double, true, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<double, true, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && transpose_b)
    {
        matrix_multiplication_seq<double, false, true>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
    if (!transpose_a && !transpose_b)
    {
        matrix_multiplication_seq<double, false, false>(static_cast<linalg_int>(rows),
            static_cast<linalg_int>(columns),
            static_cast<linalg_int>(depth),
            a,
            static_cast<linalg_int>(lda),
            b,
            static_cast<linalg_int>(ldb),
            c,
            static_cast<linalg_int>(ldc));
    }
}

#undef A
#undef B
#undef C

#endif

}  // namespace detail

void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    float const*                a,
    linalg_int                lda,
    float const*                b,
    linalg_int                ldb,
    float*                      c,
    linalg_int                ldc)
{
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "matrix_multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::matmul_mkl_f32(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#elif defined(LINALG_ENABLE_BLAS)
    detail::matmul_blas_f32(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#else
    detail::matmul_scalar_f32(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#endif
}

void matrix_multiplication(bool transpose_a,
    bool                        transpose_b,
    linalg_int                rows,
    linalg_int                columns,
    linalg_int                depth,
    double const*               a,
    linalg_int                lda,
    double const*               b,
    linalg_int                ldb,
    double*                     c,
    linalg_int                ldc)
{
    LOGGING_CHECK(a != nullptr && b != nullptr && c != nullptr, "matrix_multiplication: a, b, and c must not be null");
    LOGGING_CHECK(rows > 0 && columns > 0 && depth > 0 && lda > 0 && ldb > 0 && ldc > 0, "matrix_multiplication: rows/columns/depth/lda/ldb/ldc must be positive");
#if defined(LINALG_ENABLE_MKL)
    detail::matmul_mkl_f64(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#elif defined(LINALG_ENABLE_BLAS)
    detail::matmul_blas_f64(transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#else
    detail::matmul_scalar_f64(
        transpose_a, transpose_b, rows, columns, depth, a, lda, b, ldb, c, ldc);
#endif
}

}  // namespace linalg
#if !defined(LINALG_ENABLE_MKL) && !defined(LINALG_ENABLE_BLAS)
#if defined(_MSC_VER) && !defined(LINALG_DISPLAY_WIN32_WARNINGS)
#pragma warning(pop)
#endif
#endif
