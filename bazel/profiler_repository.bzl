"""Expose an installed KhwarizmiAnalytix/Profiler package without vendoring it.

Mirrors rocm_repository.bzl's approach for an installed SDK: CMake resolves
Profiler via `find_package(Profiler CONFIG REQUIRED)` against
CMAKE_PREFIX_PATH (see CMakeLists.txt's LINALG_ENABLE_PROFILING block); there
is no ThirdParty/ submodule for it to vendor instead. Only fetched when a
profiling target is actually requested (see Testing/Profile/BUILD.bazel's
target_compatible_with), so a plain `bazel build //...` never needs this
repo or PROFILER_PATH set.
"""

def _profiler_repository_impl(ctx):
    prefix = ctx.os.environ.get("PROFILER_PATH")
    if not prefix:
        fail(
            "LINALG_ENABLE_PROFILING is on but PROFILER_PATH is not set. Build " +
            "and install KhwarizmiAnalytix/Profiler, then pass " +
            "--repo_env=PROFILER_PATH=/path/to/its/install/prefix.",
        )
    sdk = ctx.path(prefix)
    if not sdk.get_child("include/profiler.h").exists:
        fail("Profiler package not found at %s/include/profiler.h" % prefix)
    ctx.symlink(sdk.get_child("include"), "include")
    lib = sdk.get_child("lib")
    if not lib.exists:
        lib = sdk.get_child("lib64")
    if not lib.exists:
        fail(
            "Profiler library directory not found under " + prefix +
            ". Expected lib/ or lib64/ to exist in the installed Profiler package.",
        )
    ctx.symlink(lib, "lib")
    ctx.file("BUILD.bazel", '''
load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "profiler",
    hdrs = glob(["include/**"]),
    srcs = glob(
        ["lib/libprofiler.*", "lib/libProfiler.*"],
        allow_empty = True,
    ),
    includes = ["include"],
)
''')

profiler_repository = repository_rule(
    implementation = _profiler_repository_impl,
    environ = ["PROFILER_PATH"],
    local = True,
)
