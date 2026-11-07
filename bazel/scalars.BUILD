load("@rules_cc//cc:defs.bzl", "cc_library")
load("@xsigma_linearalgebra//bazel:linear_algebra.bzl", "linear_algebra_arch_copts", "linear_algebra_gpu_copts")

# Match Scalars' one-hot CMake backend guards (AVX2 also enables AVX).
# These labels belong to LinearAlgebra, not this external Scalars repository.
cc_library(
    name = "scalars",
    srcs = [
        "src/core/scalar_type.cpp",
        "src/math/complex_math.cpp",
        "src/math/normal_cdf.cpp",
    ],
    hdrs = glob(["include/**/*.h"]),
    copts = linear_algebra_arch_copts() + linear_algebra_gpu_copts(),
    defines = [
        "SCALARS_STATIC_DEFINE",
        "SCALARS_HAS_SVML=0",
        "SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS=0",
        "SCALARS_HAS_MKL=0",
        "SCALARS_HAS_SLEEF=0",
        "SCALARS_HAS_ACCELERATE=0",
        "SCALARS_PACKET_SIZE=1",
        "SCALARS_HAS_METAL=0",
    ] + select({
        "@xsigma_linearalgebra//bazel:linalg_cpu_sse": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=1",
            "SCALARS_HAS_AVX=0",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=0",
        ],
        "@xsigma_linearalgebra//bazel:linalg_cpu_avx": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=1",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=0",
        ],
        "@xsigma_linearalgebra//bazel:linalg_cpu_avx2": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=1",
            "SCALARS_HAS_AVX2=1",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=0",
            "SCALARS_CPU_CAPABILITY_AVX2",
        ],
        "@xsigma_linearalgebra//bazel:linalg_cpu_avx512": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=0",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=1",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=0",
            "SCALARS_CPU_CAPABILITY_AVX512",
        ],
        "@xsigma_linearalgebra//bazel:linalg_cpu_neon": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=0",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=1",
            "SCALARS_HAS_SVE=0",
        ],
        "@xsigma_linearalgebra//bazel:linalg_cpu_sve": [
            "SCALARS_VECTORIZED=1",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=0",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=1",
        ],
        "//conditions:default": [
            "SCALARS_VECTORIZED=0",
            "SCALARS_HAS_SSE=0",
            "SCALARS_HAS_AVX=0",
            "SCALARS_HAS_AVX2=0",
            "SCALARS_HAS_AVX512=0",
            "SCALARS_HAS_NEON=0",
            "SCALARS_HAS_SVE=0",
        ],
    }) + select({
        "@xsigma_linearalgebra//bazel:linalg_gpu_cuda": [
            "SCALARS_HAS_CUDA=1",
            "SCALARS_HAS_HIP=0",
        ],
        "@xsigma_linearalgebra//bazel:linalg_gpu_hip": [
            "SCALARS_HAS_CUDA=0",
            "SCALARS_HAS_HIP=1",
        ],
        "//conditions:default": [
            "SCALARS_HAS_CUDA=0",
            "SCALARS_HAS_HIP=0",
        ],
    }),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = select({
        "@xsigma_linearalgebra//bazel:linalg_gpu_cuda": ["@rules_cuda//cuda:runtime"],
        "@xsigma_linearalgebra//bazel:linalg_gpu_hip": ["@local_config_hip//:hip"],
        "//conditions:default": [],
    }),
)
