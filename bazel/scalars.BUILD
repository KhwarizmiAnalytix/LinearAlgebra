load("@rules_cc//cc:defs.bzl", "cc_library")

# Bazel's native backend currently uses scalar loops. CMake additionally
# configures Scalars' CPU SIMD backends through LINALG_VECTORIZATION_ISA.
cc_library(
    name = "scalars",
    srcs = ["src/core/scalar_type.cpp", "src/math/complex_math.cpp", "src/math/normal_cdf.cpp"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    defines = ["SCALARS_STATIC_DEFINE", "SCALARS_VECTORIZED=0"],
    visibility = ["//visibility:public"],
)
