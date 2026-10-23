/*
 * LinearAlgebra — compiler portability and shared numerical source macros.
 */
#pragma once

// Marks an intentionally-unused function parameter (e.g. `pivot` in the
// non-pivoting fallback paths of lu_decomposition / linear_solver /
// inversion). Placed as a leading attribute-specifier-seq on the
// parameter declaration, so plain [[maybe_unused]] is sufficient in C++17.
#define LINALG_UNUSED [[maybe_unused]]

// Shared numerical helpers execute on the host in C++ and on the device
// when compiled by CUDA/HIP. Metal's qualifiers are supplied in its source
// preamble below because its numerical code is compiled at runtime.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LINALG_INLINE __device__ inline
#else
#define LINALG_INLINE inline
#endif
#define LINALG_ADDRESS

// Objective-C++ embeds the portable numerical bodies as Metal source.
// Stringify directly so host macros such as NAN cannot expand into compiler
// builtins unsupported by Metal. C++/CUDA/HIP compile the same bodies directly.
#if defined(__OBJC__)
#define LINALG_SCHUR_CORE(...) #__VA_ARGS__
#define LINALG_SQUARE_ROOT_CORE(...) #__VA_ARGS__
#define LINALG_LOGARITHM_CORE(...) #__VA_ARGS__
#else
#define LINALG_SCHUR_CORE(...) __VA_ARGS__
#define LINALG_SQUARE_ROOT_CORE(...) __VA_ARGS__
#define LINALG_LOGARITHM_CORE(...) __VA_ARGS__
#endif

#define LINALG_METAL_CORE_PREAMBLE                                                                 \
    "#include <metal_stdlib>\n"                                                                    \
    "using namespace metal;\n"                                                                     \
    "#define LINALG_INLINE inline\n"                                                               \
    "#define LINALG_ADDRESS device\n"
