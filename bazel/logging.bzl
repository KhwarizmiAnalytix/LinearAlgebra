# =============================================================================
# LinearAlgebra — compatibility shim for ThirdParty/Logging's Bazel package
# =============================================================================
# ThirdParty/Logging is vendored as a plain subdirectory of this workspace
# (matching the #include "ThirdParty/Logging/..." paths baked into
# include/matrix_operation/*.cxx), not a separate Bazel repository. Its own
# BUILD.bazel files load their compile-option helpers via the absolute label
# "//bazel:logging.bzl" — evaluated inside this (the only) workspace, that
# label resolves to this file rather than Logging's own bazel/logging.bzl.
#
# Re-export the real implementation instead of duplicating it, so upstream
# changes to Logging's own helpers keep applying automatically; per
# .augment/rules/ThirdParty.md, vendored sources are never edited in place.
# =============================================================================

load(
    "//ThirdParty/Logging/bazel:logging.bzl",
    _logging_copts = "logging_copts",
    _logging_defines = "logging_defines",
    _logging_linkopts = "logging_linkopts",
)

logging_copts = _logging_copts
logging_defines = _logging_defines
logging_linkopts = _logging_linkopts
