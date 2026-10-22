#pragma once

#include <cmath>
#include <initializer_list>

namespace linalg_test
{

// Analytic references shared by CPU and GPU tests. No implementation under
// test supplies expected values, including for defective/complex spectra.
template <typename F> void for_each_matrix_function_case(F&& check)
{
    const double diagonal[]      = {0.25, 0., 0., 0., 1., 0., 0., 0., 16.};
    const double diagonal_root[] = {0.5, 0., 0., 0., 1., 0., 0., 0., 4.};
    const double diagonal_log[]  = {std::log(0.25), 0., 0., 0., 0., 0., 0., 0., std::log(16.)};
    check(3, diagonal, diagonal_root, false);
    check(3, diagonal, diagonal_log, true);

    const double zero[9]    = {};
    const double identity[] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    check(3, zero, zero, false);
    check(3, identity, identity, false);
    check(3, identity, zero, true);

    const double jordan[]      = {4., 2., 0., 0., 4., 2., 0., 0., 4.};
    const double jordan_root[] = {2., 0.5, -0.0625, 0., 2., 0.5, 0., 0., 2.};
    const double jordan_log[]  = {
        std::log(4.), 0.5, -0.125, 0., std::log(4.), 0.5, 0., 0., std::log(4.)};
    check(3, jordan, jordan_root, false);
    check(3, jordan, jordan_log, true);

    const double spd[]      = {5., 4., 1., 4., 6., 4., 1., 4., 5.};
    const double spd_root[] = {2., 1., 0., 1., 2., 1., 0., 1., 2.};
    check(3, spd, spd_root, false);

    const double rotation[]      = {0., -1., 1., 0.};
    const double c               = std::sqrt(0.5);
    const double rotation_root[] = {c, -c, c, c};
    const double angle           = std::acos(-1.) / 2.;
    const double rotation_log[]  = {0., -angle, angle, 0.};
    check(2, rotation, rotation_root, false);
    check(2, rotation, rotation_log, true);

    // Similarity produces a dense, nonsymmetric matrix with a conjugate pair.
    const double p[]          = {1., 1., 0., 0., 1., 1., 1., 0., 1.};
    const double inverse[]    = {0.5, -0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5};
    const double block[]      = {0., -1., 0., 1., 0., 0., 0., 0., 4.};
    const double block_root[] = {c, -c, 0., c, c, 0., 0., 0., 2.};
    const double block_log[]  = {0., -angle, 0., angle, 0., 0., 0., 0., std::log(4.)};
    auto         transform    = [&](const double* a, double* result)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                result[i * 3 + j] = 0.;
                for (int k = 0; k < 3; ++k)
                {
                    for (int l = 0; l < 3; ++l)
                    {
                        result[i * 3 + j] += p[i * 3 + k] * a[k * 3 + l] * inverse[l * 3 + j];
                    }
                }
            }
        }
    };
    double input[9], expected[9];
    transform(block, input);
    transform(block_root, expected);
    check(3, input, expected, false);
    transform(block_log, expected);
    check(3, input, expected, true);
    transform(jordan, input);
    transform(jordan_root, expected);
    check(3, input, expected, false);
    transform(jordan_log, expected);
    check(3, input, expected, true);

    const double tiny            = std::ldexp(1., -20);
    const double near_identity[] = {1. + tiny, 0.001, 0., 1. + tiny};
    const double near_log[]      = {std::log1p(tiny), 0.001 / (1. + tiny), 0., std::log1p(tiny)};
    const double near_root[]     = {
        std::sqrt(1. + tiny), 0.001 / (2. * std::sqrt(1. + tiny)), 0., std::sqrt(1. + tiny)};
    check(2, near_identity, near_log, true);
    check(2, near_identity, near_root, false);

    const double extreme[]      = {1.e-20, 0., 0., 1.e20};
    const double extreme_root[] = {1.e-10, 0., 0., 1.e10};
    const double extreme_log[]  = {std::log(1.e-20), 0., 0., std::log(1.e20)};
    check(2, extreme, extreme_root, false);
    check(2, extreme, extreme_log, true);

    // A larger dense similarity exercises Hessenberg reduction and deflation.
    // P = I + u*v^T, u_i=0.2, v_i=(-1)^i*0.1, so v^T*u=0 and
    // P^{-1}=I-u*v^T. This gives independent analytic matrix functions.
    for (bool is_log : {false, true})
    {
        double dense[36], reference[36];
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                dense[i * 6 + j]     = 0.;
                reference[i * 6 + j] = 0.;
                for (int k = 0; k < 6; ++k)
                {
                    const double p_ik  = (i == k ? 1. : 0.) + (k % 2 == 0 ? 0.02 : -0.02);
                    const double pi_kj = (k == j ? 1. : 0.) - (j % 2 == 0 ? 0.02 : -0.02);
                    dense[i * 6 + j] += p_ik * (k + 1.) * pi_kj;
                    reference[i * 6 + j] +=
                        p_ik * (is_log ? std::log(k + 1.) : std::sqrt(k + 1.)) * pi_kj;
                }
            }
        }
        check(6, dense, reference, is_log);
    }
}

}  // namespace linalg_test
