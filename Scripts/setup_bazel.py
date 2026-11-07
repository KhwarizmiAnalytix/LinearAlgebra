#!/usr/bin/env python3
"""LinearAlgebra Bazel Build Configuration Script.

Design follows KhwarizmiAnalytix/XSigma's Scripts/setup_bazel.py (dotted-token
CLI, BazelConfiguration class, the llvm-profdata/llvm-cov coverage pipeline
that works around Bazel's own broken C++ coverage merging), wired to every
--define= key bazel/BUILD.bazel's config_settings actually consume, for
parity with .github/scripts/flag_matrix.py's CMake option inventory
(domains() + FOCUSED). Each flag_matrix.py key lands in one of three buckets
here — see print_help()'s "Flag parity with flag_matrix.py" section for the
full breakdown:

  1. Source-gating --define= keys, wired straight to bazel/BUILD.bazel
     config_settings (these change which .cpp/.cu/.hip files glob() selects,
     or flip a LINALG_HAS_* define): linalg_enable_mkl, linalg_enable_blas
     (also covers LINALG_ENABLE_BLIS — Bazel has no separate BLIS
     config_setting; see "blis" below), linalg_lu_pivoting,
     linalg_display_win32_warnings, linalg_gpu_backend (cuda/hip; CUDA SM /
     HIP gfx architectures via the existing rules_cuda extension flag and
     --copt=--offload-arch=), linalg_cpu_backend, linalg_enable_profiling
     (Testing/Profile; see bazel/profiler_repository.bzl).
  2. Compiler/linker-level flags with no source-level branching — the
     idiomatic Bazel way to express these is --copt/--linkopt/--repo_env on
     the command line, not a custom --define + select(): sanitizer.*, lto.*,
     linker.*, cache[.ccache|.sccache|.buildcache], icecc, minsizerel.
  3. Genuinely CMake-only in this repo (no Bazel equivalent at all): the
     developer-tool switches (clangtidy/fix/iwyu/spell/valgrind/cppcheck),
     LINALG_BUILD_EXAMPLES (no examples/BUILD.bazel), LINALG_GPU_BACKEND=metal
     (no Bazel Metal toolchain here), LINALG_USE_SYSTEM_LOGGING/SCALARS and
     LINALG_ENABLE_EXTERNAL (MODULE.bazel always vendors via
     local_path_override; there is no "resolve from an installed package"
     path in bzlmod here), and the CMake generator/BUILD_TESTING granularity
     (Bazel always builds the test target graph; `test` vs `build` already
     controls whether it's *run*).

Usage:
    python Scripts/setup_bazel.py config.build.test
    python Scripts/setup_bazel.py build.test.release
    python Scripts/setup_bazel.py test --backend.glog
    python Scripts/setup_bazel.py coverage
"""

import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Optional

try:
    import colorama

    colorama.init()
    COLOR_CYAN = colorama.Fore.CYAN
    COLOR_GREEN = colorama.Fore.GREEN
    COLOR_RED = colorama.Fore.RED
    COLOR_WHITE = colorama.Fore.WHITE
    COLOR_YELLOW = colorama.Fore.YELLOW
    COLOR_RESET = colorama.Style.RESET_ALL
except ImportError:  # Bazel CI jobs may not pip-install colorama
    COLOR_CYAN = ""
    COLOR_GREEN = ""
    COLOR_RED = ""
    COLOR_WHITE = ""
    COLOR_YELLOW = ""
    COLOR_RESET = ""


def print_status(message: str, status: str = "INFO") -> None:
    colors = {
        "INFO": COLOR_CYAN,
        "SUCCESS": COLOR_GREEN,
        "WARNING": COLOR_YELLOW,
        "ERROR": COLOR_RED,
    }
    print(f"{colors.get(status, COLOR_WHITE)}[{status}]{COLOR_RESET} {message}")


def _find_bazel_executable() -> Optional[str]:
    """Return the absolute path of bazelisk or bazel, or None."""
    for cmd in ["bazelisk", "bazel"]:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def get_bazel_command() -> str:
    cmd = _find_bazel_executable()
    if cmd:
        return cmd
    raise RuntimeError("Neither bazel nor bazelisk found in PATH")


def bazel_prefix() -> list[str]:
    """Argv prefix that can spawn Bazel on every platform.

    Windows ``.cmd``/``.bat`` launchers need ``cmd /c``; CreateProcess
    cannot execute them directly (WinError 2).
    """
    exe = get_bazel_command()
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", exe]
    return [exe]


def _repo_root() -> str:
    """This repo's root — Scripts/setup_bazel.py's parent directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shell_single_quote(part: str) -> str:
    """Quote one POSIX shell word, including embedded single quotes."""
    return "'" + part.replace("'", "'\"'\"'") + "'"


def _sha256_file(path: str) -> str:
    """Hash a launcher input so a rewritten wrapper changes the cache key."""
    if not os.path.isfile(path):
        return "unresolved"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_executable(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


_SANITIZER_NAMES = {"address", "undefined", "thread", "memory", "leak"}
_LTO_MODES = {"off", "thin", "full", "ipo", "auto"}
_LINKER_CHOICES = {"default", "lld", "mold", "gold", "lld-link"}
_CACHE_BACKENDS = {"ccache", "sccache", "buildcache", "none"}
_CPU_BACKENDS = {"no", "sse", "avx", "avx2", "avx512", "neon", "sve"}
_GPU_BACKENDS = {"cuda", "hip", "metal", "none"}
_BACKEND_NAMES = {"native", "loguru", "glog", "spdlog"}


def _merge_dotted_segments(parts: list[str]) -> list[str]:
    """Merge split segments like backend.glog, sanitizer.address into single tokens."""
    out: list[str] = []
    pl = [p.lower() for p in parts]
    i = 0
    while i < len(pl):
        if pl[i] == "backend" and i + 1 < len(pl) and pl[i + 1] in _BACKEND_NAMES:
            out.append(pl[i + 1])
            i += 2
        elif (
            pl[i] == "sanitizer"
            and i + 1 < len(pl)
            and pl[i + 1]
            and all(t in _SANITIZER_NAMES for t in pl[i + 1].split(","))
        ):
            # pl[i + 1] may itself be a comma-combo (e.g. "address,undefined"),
            # which .split(".") never breaks apart since "," isn't a dot.
            out.append(f"sanitizer.{pl[i + 1]}")
            i += 2
        elif pl[i] == "lto" and i + 1 < len(pl) and pl[i + 1] in _LTO_MODES:
            out.append(f"lto.{pl[i + 1]}")
            i += 2
        elif pl[i] == "linker" and i + 1 < len(pl):
            out.append(f"linker.{pl[i + 1]}")
            i += 2
        elif pl[i] == "cache" and i + 1 < len(pl) and pl[i + 1] in _CACHE_BACKENDS:
            out.append(f"cache.{pl[i + 1]}")
            i += 2
        elif pl[i] == "cpu" and i + 1 < len(pl) and pl[i + 1] in _CPU_BACKENDS:
            out.append(f"cpu.{pl[i + 1]}")
            i += 2
        elif pl[i] == "gpu" and i + 1 < len(pl) and pl[i + 1] in _GPU_BACKENDS:
            out.append(f"gpu.{pl[i + 1]}")
            i += 2
        elif pl[i] in ("cudaarch", "hiparch") and i + 1 < len(pl):
            # Architecture lists are comma-separated (e.g. sm_80,sm_86), so the
            # value itself never contains a further "." to re-split on.
            out.append(f"{pl[i]}.{pl[i + 1]}")
            i += 2
        else:
            out.append(pl[i])
            i += 1
    return out


class LinearAlgebraBazelConfiguration:
    """Manages Bazel build configuration and execution for LinearAlgebra."""

    def __init__(self, args: list[str]) -> None:
        self.args = args
        self.build_type = "debug"
        self.cxx_standard: Optional[str] = None
        self.targets: list[str] = ["//..."]
        self.run_tests = False
        self.run_build = False
        self.run_clean = False
        self.run_config = False
        self.run_coverage = False
        self.timing_data: dict[str, float] = {}
        self.slow_phase_warn_seconds: float = 60.0
        self.use_batch: bool = False
        self.verbose_tests = False
        self.subprocess_timeout: int = 600
        self.system = platform.system()

        # Compiler: None = Bazel's autoconfigured default toolchain (unlike CMake,
        # switching this only takes effect via --repo_env=CC=/CXX=, which forces
        # Bazel to re-run its C/C++ autoconfiguration).
        self.compiler: Optional[str] = None
        self.compiler_c: Optional[str] = None
        self.compiler_cxx: Optional[str] = None

        # Mirrors CMake LOGGING_BACKEND (native | loguru | glog | spdlog); None =
        # bazel/BUILD.bazel's own default (loguru, via the vendored xsigma_logging
        # module's config_settings — --define is global, so this still reaches it).
        self.logging_backend: Optional[str] = None
        self.shared_libs: bool = False
        self.disable_magic_enum: bool = False
        self.disable_cxa_demangle: bool = False
        self.portable_float_format: bool = False
        self.std_format: bool = False
        self.exception_mode: Optional[str] = None  # None | "log_fatal"

        # LinearAlgebra-specific --define= keys (bazel/BUILD.bazel config_settings).
        self.enable_mkl: bool = False
        self.enable_blas: bool = False
        self.blis_label: bool = False  # display-only; "blis" maps onto enable_blas
        self.lu_pivoting: bool = False
        self.win32_warnings: bool = False
        self.gpu_backend: Optional[str] = None  # None | "cuda" | "hip"
        self.cuda_archs: Optional[str] = None  # e.g. "sm_80,sm_86"
        self.hip_archs: Optional[list[str]] = None  # e.g. ["gfx90a", "gfx1100"]
        self.cpu_backend: Optional[str] = None
        self.enable_profiling: bool = False

        # Compiler/linker-level flags (--copt/--linkopt/--repo_env, no --define).
        self.sanitizer: Optional[str] = None
        self.lto_mode: Optional[str] = None
        self.linker: Optional[str] = None
        self.cache_backend: Optional[str] = None  # "ccache" | "sccache" | "buildcache"
        self.icecc: bool = False

        # CMake-only flags — not wired to any Bazel --config/--define/--copt in
        # this repo's bazel/BUILD.bazel or .bazelrc; tracked only so the
        # summary/warnings are accurate. See module docstring, bucket 3.
        self.spell: bool = False
        self.clangtidy: bool = False
        self.fix: bool = False
        self.iwyu: bool = False
        self.valgrind: bool = False
        self.examples: bool = False
        self.cppcheck: bool = False
        self.metal: bool = False
        self.use_system_logging: bool = False
        self.use_system_scalars: bool = False
        self.enable_external: bool = False

        self._parse_arguments()
        self._apply_defaults()

    def _cmake_only(self, token: str, attr: Optional[str] = None) -> None:
        if attr:
            setattr(self, attr, True)
        print_status(
            f"Token '{token}': CMake-only (Scripts/setup.py) — not wired to a Bazel "
            "--config/--define/--copt in this repo's bazel/BUILD.bazel or .bazelrc.",
            "WARNING",
        )

    def _parse_arguments(self) -> None:
        for arg in self.args:
            arg_lower = arg.lower()

            if (
                arg_lower in ("clang", "gcc")
                or "clang" in arg_lower
                or "gcc" in arg_lower
                or "g++" in arg_lower
            ):
                if arg_lower not in (
                    "clangtidy",
                    "clang-tidy",
                    "clang_tidy",
                    "cppcheck",
                ):
                    self._set_compiler(arg_lower)
                    continue

            if arg_lower in ["debug", "release", "relwithdebinfo", "minsizerel"]:
                self.build_type = arg_lower
            elif arg_lower in ["cxx20", "cxx23"]:
                self.cxx_standard = arg_lower
            elif arg_lower in _BACKEND_NAMES:
                self.logging_backend = arg_lower
            elif arg_lower.startswith("backend."):
                backend = arg_lower.split(".", 1)[1]
                if backend in _BACKEND_NAMES:
                    self.logging_backend = backend
                else:
                    print_status(
                        f"Invalid logging backend '{backend}'. Use native, loguru, "
                        "glog, or spdlog.",
                        "ERROR",
                    )
                    sys.exit(1)
            elif arg_lower in ("shared", "static"):
                # CMake's BUILD_SHARED_LIBS defaults ON for this repo; Bazel's own
                # default (no define) is static, so both spellings are accepted and
                # only "static" (or omitting the token) keeps Bazel's real default.
                self.shared_libs = arg_lower == "shared"
            elif arg_lower in ("nomagicenum", "no_magicenum"):
                self.disable_magic_enum = True
            elif arg_lower in ("nocxademangle", "no_cxademangle"):
                self.disable_cxa_demangle = True
            elif arg_lower == "portablefloat":
                self.portable_float_format = True
            elif arg_lower == "stdformat":
                self.std_format = True
            elif arg_lower == "logfatal":
                self.exception_mode = "log_fatal"
            elif arg_lower == "throw":
                self.exception_mode = None
            elif arg_lower == "mkl":
                self.enable_mkl = True
            elif arg_lower == "blas":
                self.enable_blas = True
            elif arg_lower == "blis":
                # LINALG_ENABLE_BLIS activates the same blas_lapacke source
                # selection as LINALG_ENABLE_BLAS (see CMakeLists.txt); the only
                # difference is *which* library find_package() resolves, and
                # Bazel never does that resolution for either (both already rely
                # on the toolchain's default search paths — see BUILD.bazel's
                # linalg_enable_mkl/linalg_enable_blas comment). No separate
                # Bazel config_setting is needed.
                self.enable_blas = True
                self.blis_label = True
            elif arg_lower in ("lupivoting", "lu_pivoting", "pivoting"):
                self.lu_pivoting = True
            elif arg_lower in ("win32warnings", "displaywin32warnings"):
                self.win32_warnings = True
            elif arg_lower in ("cuda", "hip"):
                self.gpu_backend = arg_lower
            elif arg_lower == "metal":
                self._cmake_only(arg_lower, "metal")
            elif arg_lower.startswith("gpu."):
                value = arg_lower.split(".", 1)[1]
                if value == "none":
                    self.gpu_backend = None
                elif value == "metal":
                    self._cmake_only(arg_lower, "metal")
                elif value in ("cuda", "hip"):
                    self.gpu_backend = value
                else:
                    print_status(f"Invalid GPU backend '{value}'.", "ERROR")
                    sys.exit(1)
            elif arg_lower.startswith("cudaarch."):
                self.cuda_archs = arg_lower.split(".", 1)[1]
            elif arg_lower.startswith("hiparch."):
                self.hip_archs = arg_lower.split(".", 1)[1].split(",")
            elif arg_lower in ("no", "sse", "avx", "avx2", "avx512", "neon", "sve"):
                self.cpu_backend = arg_lower
            elif arg_lower == "cpuno":
                self.cpu_backend = "no"
            elif arg_lower.startswith("cpu."):
                value = arg_lower.split(".", 1)[1]
                if value in _CPU_BACKENDS:
                    self.cpu_backend = value
                else:
                    print_status(f"Invalid CPU backend '{value}'.", "ERROR")
                    sys.exit(1)
            elif arg_lower == "profiling":
                self.enable_profiling = True
            elif arg_lower.startswith("sanitizer."):
                self.sanitizer = arg_lower.split(".", 1)[1]
            elif arg_lower.startswith("lto"):
                self.lto_mode = (
                    arg_lower.split(".", 1)[1] if "." in arg_lower else "auto"
                )
            elif arg_lower.startswith("linker."):
                self.linker = arg_lower.split(".", 1)[1]
            elif arg_lower in ("ccache", "sccache", "buildcache"):
                self.cache_backend = arg_lower
            elif arg_lower.startswith("cache."):
                backend = arg_lower.split(".", 1)[1]
                if backend in _CACHE_BACKENDS:
                    self.cache_backend = None if backend == "none" else backend
                else:
                    print_status(f"Invalid cache backend '{backend}'.", "ERROR")
                    sys.exit(1)
            elif arg_lower == "cache":
                # Mirrors CMake: LINALG_ENABLE_CACHE defaults ON but
                # LINALG_CACHE_BACKEND defaults "none", i.e. a no-op until a
                # backend is actually named.
                print_status(
                    "Token 'cache': no backend named (mirrors CMake's "
                    "LINALG_CACHE_BACKEND=none default — a no-op). Use "
                    "cache.ccache, cache.sccache, or cache.buildcache.",
                    "WARNING",
                )
            elif arg_lower == "icecc":
                if self.system == "Windows":
                    print_status(
                        "icecc: not offered on Windows (matches "
                        ".github/scripts/flag_matrix.py's LINALG_ENABLE_ICECC domain).",
                        "ERROR",
                    )
                    sys.exit(1)
                self.icecc = True
            elif arg_lower == "examples":
                self._cmake_only(arg_lower, "examples")
            elif arg_lower == "cppcheck":
                self._cmake_only(arg_lower, "cppcheck")
            elif arg_lower == "spell":
                self._cmake_only(arg_lower, "spell")
            elif arg_lower in ("clangtidy", "clang-tidy", "clang_tidy"):
                self._cmake_only(arg_lower, "clangtidy")
            elif arg_lower == "fix":
                self._cmake_only(arg_lower, "fix")
            elif arg_lower == "iwyu":
                self._cmake_only(arg_lower, "iwyu")
            elif arg_lower == "valgrind":
                self._cmake_only(arg_lower, "valgrind")
            elif arg_lower in ("usesystemlogging", "use_system_logging"):
                self._cmake_only(arg_lower, "use_system_logging")
            elif arg_lower in ("usesystemscalars", "use_system_scalars"):
                self._cmake_only(arg_lower, "use_system_scalars")
            elif arg_lower in ("enableexternal", "enable_external"):
                self._cmake_only(arg_lower, "enable_external")
            elif arg_lower == "vv":
                self.verbose_tests = True
            elif arg_lower == "batch":
                self.use_batch = True
            elif arg_lower == "build":
                self.run_build = True
            elif arg_lower == "test":
                self.run_tests = True
            elif arg_lower == "coverage":
                self.run_coverage = True
            elif arg_lower == "clean":
                self.run_clean = True
            elif arg_lower == "config":
                self.run_config = True

    def _set_compiler(self, arg: str) -> None:
        if "clang" in arg and arg not in ("clang-cl",):
            self.compiler = "clang"
            self.compiler_c = arg
            self.compiler_cxx = arg.replace("clang", "clang++")
        elif "gcc" in arg or "g++" in arg:
            self.compiler = "gcc"
            if "g++" in arg:
                self.compiler_cxx = arg
                self.compiler_c = arg.replace("g++", "gcc")
            else:
                self.compiler_c = arg
                self.compiler_cxx = arg.replace("gcc", "g++")

    def _apply_defaults(self) -> None:
        if not self.compiler:
            self.compiler = "clang"
            self.compiler_c = "clang"
            self.compiler_cxx = "clang++"
        print_status(f"Using default compiler: {self.compiler.upper()}", "INFO")

        if self.sanitizer:
            self._validate_sanitizer()
        if self.icecc and self.system == "Windows":
            print_status("icecc: not available on Windows; ignoring.", "WARNING")
            self.icecc = False
        if self.cache_backend and self.system == "Windows":
            print_status(
                f"cache.{self.cache_backend}: Bazel's autoconfigured MSVC toolchain "
                "doesn't go through --repo_env=CC/CXX the way Clang/GCC do, so this "
                "helper can't wrap it with a compiler-cache launcher on Windows; "
                "ignoring.",
                "WARNING",
            )
            self.cache_backend = None
        if self.icecc and self.compiler == "gcc" and self.system == "Darwin":
            print_status(
                "icecc: GCC cross-compilation under icecc on macOS is not "
                "validated by this helper; proceeding anyway.",
                "WARNING",
            )

    def _validate_sanitizer(self) -> None:
        """Mirror Cmake/sanitize.cmake's compiler/platform/combination checks."""
        assert self.sanitizer is not None
        types = self.sanitizer.split(",")
        for t in types:
            if t not in _SANITIZER_NAMES:
                print_status(f"Unsupported sanitizer: {t}", "ERROR")
                sys.exit(1)
        if "address" in types and ("thread" in types or "memory" in types):
            print_status(
                f"Incompatible sanitizer combination: {self.sanitizer}", "ERROR"
            )
            sys.exit(1)
        if self.compiler not in ("clang", "gcc"):
            print_status("Sanitizers require GCC or Clang.", "ERROR")
            sys.exit(1)
        if self.system == "Windows":
            if types != ["address"]:
                print_status(
                    "Windows only supports sanitizer.address in this repo's CI "
                    "matrix (.github/scripts/flag_matrix.py).",
                    "ERROR",
                )
                sys.exit(1)
        elif self.system == "Darwin":
            if any(t in ("thread", "memory", "leak") for t in types):
                print_status(
                    f"macOS does not support sanitizer type(s) in '{self.sanitizer}' "
                    "(.github/scripts/flag_matrix.py restricts macOS to "
                    "address/undefined).",
                    "ERROR",
                )
                sys.exit(1)
        elif "memory" in types and self.compiler != "clang":
            print_status("sanitizer.memory requires Clang.", "ERROR")
            sys.exit(1)

    def _resolve_lto_flags(self) -> list[str]:
        """Approximate Cmake/lto.cmake's mode resolution as raw copt/linkopt.

        Bazel has no per-target INTERPROCEDURAL_OPTIMIZATION property, so
        "ipo" and the msvc leg of "auto" are approximated with /GL + /LTCG;
        everywhere else, "ipo" is treated as a synonym for "full" (both mean
        "turn on the compiler's own LTO"), and "auto" picks thin (Clang) or
        full (GCC).
        """
        mode = self.lto_mode
        if mode == "auto":
            mode = "thin" if self.compiler == "clang" else "full"
        if mode == "off" or mode is None:
            return []
        if self.system == "Windows" and self.compiler not in ("clang", "gcc"):
            return ["--copt=/GL", "--linkopt=/LTCG"]
        if mode == "thin" and self.compiler != "clang":
            print_status(
                "lto.thin requires Clang (ThinLTO); falling back to full LTO.",
                "WARNING",
            )
            mode = "full"
        flag = "-flto=thin" if mode == "thin" else "-flto"
        return [f"--copt={flag}", f"--linkopt={flag}"]

    def _resolve_linker_flags(self) -> list[str]:
        if not self.linker or self.linker == "default":
            return []
        if self.linker not in _LINKER_CHOICES:
            print_status(f"Invalid linker choice '{self.linker}'.", "ERROR")
            sys.exit(1)
        if self.linker == "lld-link":
            # CMake selects lld-link via CMAKE_LINKER on the MSVC driver; Bazel's
            # autoconfigured MSVC toolchain has no equivalent linker override here.
            print_status(
                "linker.lld-link: no Bazel equivalent for the MSVC driver in this "
                "repo; ignoring.",
                "WARNING",
            )
            return []
        return [f"--linkopt=-fuse-ld={self.linker}"]

    def _write_compiler_launcher(
        self, real_compiler: str, suffix: str
    ) -> Optional[str]:
        """Chain cache_backend → icecc → real_compiler, mirroring
        Cmake/icecc_cache_launcher.sh.in / icecc_compiler.sh.in.

        Returns the launcher script path, or None if neither cache nor icecc
        is active (callers should then point --repo_env=CC/CXX at
        `real_compiler` directly).

        ccache and sccache take one compiler argument and treat later bare
        paths as source files. `cache icecc clang` therefore never caches.
        When both tools are enabled, the cache runs one compiler-shaped
        wrapper and that wrapper runs Icecream around the real compiler.
        """
        if not self.cache_backend and not self.icecc:
            return None
        cache_program = None
        if self.cache_backend:
            cache_program = shutil.which(self.cache_backend)
            if not cache_program:
                print_status(
                    f"cache.{self.cache_backend}: '{self.cache_backend}' not found on "
                    "PATH; install it or select a different backend.",
                    "ERROR",
                )
                sys.exit(1)
        icecc_program = None
        if self.icecc:
            icecc_program = shutil.which("icecc")
            if not icecc_program:
                print_status(
                    "icecc: 'icecc' not found on PATH; install Icecream.", "ERROR"
                )
                sys.exit(1)
        compiler = shutil.which(real_compiler) or real_compiler

        launcher_dir = os.path.join(_repo_root(), ".bazel_launchers")
        os.makedirs(launcher_dir, exist_ok=True)
        if cache_program and icecc_program:
            wrapper_path = os.path.join(launcher_dir, f"icecc_{suffix}_compiler.sh")
            _write_executable(
                wrapper_path,
                "#!/bin/sh\n"
                f"# Compiler SHA256: {_sha256_file(compiler)}\n"
                f"# Icecream SHA256: {_sha256_file(icecc_program)}\n"
                "exec "
                f"{_shell_single_quote(icecc_program)} "
                f"{_shell_single_quote(compiler)} "
                '"$@"\n',
            )
            command = [cache_program, wrapper_path]
        elif cache_program:
            command = [cache_program, compiler]
        else:
            command = [icecc_program, compiler]

        script_path = os.path.join(launcher_dir, f"launcher_{suffix}.sh")
        quoted = " ".join(_shell_single_quote(part) for part in command)
        _write_executable(script_path, f'#!/bin/sh\nexec {quoted} "$@"\n')
        return script_path

    def _config_flags(self) -> list[str]:
        """Flags that select *which build configuration* Bazel resolves.

        Shared between build_bazel_command() and _bazel_info(): `bazel info
        bazel-bin`/`output_path` resolve to a different directory per
        compilation_mode (fastbuild/dbg/opt) and per --repo_env (a different
        CC/CXX re-triggers cc_autoconf) — querying `bazel info` without these
        same flags silently returns a sibling configuration's path (e.g. the
        default fastbuild output-bin instead of the just-built dbg one),
        which then makes coverage's llvm-cov -object resolution find no
        coverage data at all.
        """
        flags: list[str] = []
        if self.build_type == "release":
            flags.extend(["-c", "opt"])
        elif self.build_type == "relwithdebinfo":
            flags.extend(["-c", "opt", "--copt=-g"])
        elif self.build_type == "minsizerel":
            msvc = self.system == "Windows" and self.compiler not in ("clang", "gcc")
            opt = "/O1" if msvc else "-Os"
            flags.extend(["-c", "opt", f"--copt={opt}"])
        else:
            flags.extend(["-c", "dbg"])

        # A cache/icecc launcher always wins over a bare --repo_env=CC/CXX for
        # the *same* real compiler, so resolve it first.
        launcher_cc = self._write_compiler_launcher(self.compiler_c or "cc", "cc")
        launcher_cxx = self._write_compiler_launcher(self.compiler_cxx or "c++", "cxx")
        if launcher_cc:
            flags.append(f"--repo_env=CC={launcher_cc}")
        elif self.compiler_c:
            flags.append(f"--repo_env=CC={self.compiler_c}")
        if launcher_cxx:
            flags.append(f"--repo_env=CXX={launcher_cxx}")
        elif self.compiler_cxx:
            flags.append(f"--repo_env=CXX={self.compiler_cxx}")

        if self.logging_backend:
            flags.append(f"--define=logging_backend={self.logging_backend}")

        if self.shared_libs:
            flags.append("--define=build_shared_libs=true")

        if self.disable_magic_enum:
            flags.append("--define=logging_enable_magic_enum=false")

        if self.disable_cxa_demangle:
            flags.append("--define=logging_enable_cxa_demangle=false")

        if self.portable_float_format:
            flags.append("--define=logging_enable_portable_float_format=true")

        if self.std_format:
            flags.append("--define=logging_format_use_std=true")

        if self.exception_mode == "log_fatal":
            flags.append("--define=logging_default_exception_mode=log_fatal")

        if self.cxx_standard:
            std_version = self.cxx_standard.replace("cxx", "")
            flags.append(f"--cxxopt=-std=c++{std_version}")
            flags.append(f"--host_cxxopt=-std=c++{std_version}")

        if self.enable_mkl:
            flags.append("--define=linalg_enable_mkl=true")

        if self.enable_blas:
            flags.append("--define=linalg_enable_blas=true")

        if self.lu_pivoting:
            flags.append("--define=linalg_lu_pivoting=true")

        if self.win32_warnings:
            flags.append("--define=linalg_display_win32_warnings=true")

        if self.gpu_backend:
            flags.append(f"--define=linalg_gpu_backend={self.gpu_backend}")
            if self.gpu_backend == "cuda" and self.cuda_archs:
                flags.append(f"--@rules_cuda//cuda:archs={self.cuda_archs}")
            if self.gpu_backend == "hip" and self.hip_archs:
                for arch in self.hip_archs:
                    flags.append(f"--copt=--offload-arch={arch}")

        if self.cpu_backend:
            flags.append(f"--define=linalg_cpu_backend={self.cpu_backend}")

        if self.enable_profiling:
            flags.append("--define=linalg_enable_profiling=true")

        # Cmake/lto.cmake skips LTO entirely under coverage/sanitizer builds;
        # mirror that rather than emitting conflicting flags.
        if self.lto_mode and not (self.sanitizer or self.run_coverage):
            flags.extend(self._resolve_lto_flags())
        elif self.lto_mode and (self.sanitizer or self.run_coverage):
            print_status(
                f"lto.{self.lto_mode}: skipped (coverage/sanitizer build; matches "
                "Cmake/lto.cmake).",
                "WARNING",
            )

        if self.sanitizer:
            flags.extend(
                [
                    "--copt=-O1",
                    "--copt=-g",
                    "--copt=-fno-omit-frame-pointer",
                    "--copt=-fno-optimize-sibling-calls",
                    f"--copt=-fsanitize={self.sanitizer}",
                    f"--linkopt=-fsanitize={self.sanitizer}",
                ]
            )

        flags.extend(self._resolve_linker_flags())

        return flags

    def build_bazel_command(
        self, action: str, extra_flags: Optional[list[str]] = None
    ) -> list[str]:
        """Build the Bazel command with all configurations.

        `extra_flags` (e.g. --test_output=...) must land before the targets:
        self.targets includes "-//..." negative patterns, and Bazel requires
        every flag to precede the "--" end-of-options marker that introduces
        those ("Negative target patterns can only appear after the end of
        options marker").
        """
        cmd = bazel_prefix()
        if self.use_batch:
            cmd.append("--batch")
        cmd.append(action)
        cmd.extend(self._config_flags())
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.append("--")
        cmd.extend(self.targets)
        return cmd

    def _kill_stale_bazel_processes(self) -> None:
        """Force-kill any stale Bazel server processes holding the output base lock."""
        print_status("Killing any stale Bazel server processes...", "INFO")
        for proc_name in ["bazel-real.exe", "bazel.exe"]:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    print_status(f"Killed stale process: {proc_name}", "WARNING")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    def _shutdown_bazel_for_batch(self) -> None:
        if not self.use_batch:
            return
        print_status(
            "Running `bazel shutdown` before `--batch` (avoids startup-option "
            "mismatch).",
            "INFO",
        )
        try:
            subprocess.run(
                bazel_prefix() + ["shutdown"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _on_off(self, condition: bool) -> str:
        return (
            f"{COLOR_GREEN}ON{COLOR_RESET}"
            if condition
            else f"{COLOR_RED}OFF{COLOR_RESET}"
        )

    def _na(self) -> str:
        return f"{COLOR_YELLOW}N/A (CMake-only){COLOR_RESET}"

    def _pf(self, label: str, value: str, width: int = 20) -> None:
        print(f"  {label:{width}}: {value}")

    def print_configuration_summary(self) -> None:
        """Print a summary of the resolved build configuration to stdout."""
        print("\n" + "=" * 80)
        print("LINEARALGEBRA BAZEL BUILD CONFIGURATION SUMMARY")
        print("=" * 80)

        print(f"\n{COLOR_CYAN}Compiler & Build Tool:{COLOR_RESET}")
        self._pf("Platform", self.system)
        self._pf("Compiler", (self.compiler or "clang").upper())
        self._pf("Build type", self.build_type.upper())
        self._pf(
            "Cxx standard",
            self.cxx_standard.replace("cxx", "C++")
            if self.cxx_standard
            else "C++20 (default)",
        )

        print(f"\n{COLOR_CYAN}LinearAlgebra backends "
              f"(real Bazel --define=):{COLOR_RESET}")
        self._pf("Shared libs", self._on_off(self.shared_libs))
        self._pf("MKL", self._on_off(self.enable_mkl))
        self._pf(
            "BLAS" + (" (via BLIS)" if self.blis_label else ""),
            self._on_off(self.enable_blas),
        )
        self._pf("LU pivoting", self._on_off(self.lu_pivoting))
        self._pf("Win32 warnings", self._on_off(self.win32_warnings))
        self._pf("GPU backend", (self.gpu_backend or "none").upper())
        if self.gpu_backend == "cuda":
            self._pf("CUDA archs", self.cuda_archs or "(compiler default)")
        if self.gpu_backend == "hip":
            archs = ",".join(self.hip_archs) if self.hip_archs else "(none set)"
            self._pf("HIP archs", archs)
        metal = self._na() if self.metal else self._on_off(False)
        self._pf("Metal GPU backend", metal)
        self._pf("CPU SIMD backend", (self.cpu_backend or "no").upper())
        self._pf("Profiling", self._on_off(self.enable_profiling))

        print(f"\n{COLOR_CYAN}Logging module (Bazel flags, forwarded to "
              f"@xsigma_logging):{COLOR_RESET}")
        self._pf("Backend", (self.logging_backend or "loguru").upper())
        self._pf("Magic enum", self._on_off(not self.disable_magic_enum))
        self._pf("Cxa demangle", self._on_off(not self.disable_cxa_demangle))
        self._pf("Portable floats", self._on_off(self.portable_float_format))
        self._pf("Std format", self._on_off(self.std_format))
        self._pf("Exception mode", (self.exception_mode or "throw").upper())

        print(f"\n{COLOR_CYAN}Compiler/linker-level flags (--copt/--linkopt/"
              f"--repo_env, no --define):{COLOR_RESET}")
        self._pf("Coverage", self._on_off(self.run_coverage))
        self._pf("Testing", self._on_off(self.run_tests))
        self._pf("Gtest", self._on_off(True))
        self._pf("Benchmark", self._on_off(True))
        sanitizer = self.sanitizer.upper() if self.sanitizer else self._on_off(False)
        self._pf("Sanitizer", sanitizer)
        self._pf("Lto", (self.lto_mode or "off").upper())
        self._pf("Linker", self.linker or "default")
        self._pf("Cache backend", (self.cache_backend or "none").upper())
        self._pf("Icecc", self._on_off(self.icecc))

        print(f"\n{COLOR_CYAN}CMake-only (Scripts/setup.py), no Bazel equivalent "
              f"in this repo:{COLOR_RESET}")
        self._pf("Examples", self._na() if self.examples else self._on_off(False))
        self._pf("Clang-tidy", self._na() if self.clangtidy else self._on_off(False))
        self._pf("Fix", self._na() if self.fix else self._on_off(False))
        self._pf("Iwyu", self._na() if self.iwyu else self._on_off(False))
        self._pf("Cppcheck", self._na() if self.cppcheck else self._on_off(False))
        self._pf("Spell", self._na() if self.spell else self._on_off(False))
        self._pf("Valgrind", self._na() if self.valgrind else self._on_off(False))
        use_sys_log = self._na() if self.use_system_logging else self._on_off(False)
        self._pf("Use system logging", use_sys_log)
        use_sys_scalars = self._na() if self.use_system_scalars else self._on_off(False)
        self._pf("Use system scalars", use_sys_scalars)
        enable_ext = self._na() if self.enable_external else self._on_off(False)
        self._pf("Enable external", enable_ext)

        if self.run_build or self.run_tests or self.run_coverage:
            action = (
                "test"
                if self.run_tests
                else ("coverage" if self.run_coverage else "build")
            )
            cmd = self.build_bazel_command(action)
            print(f"\n{COLOR_CYAN}Bazel Command:{COLOR_RESET}")
            print(f"  {' '.join(cmd)}")

        print("\n" + "=" * 80 + "\n")

    def config(self) -> None:
        """Handle config action (summary already printed by execute())."""

    def clean(self) -> None:
        if not self.run_clean:
            return
        print_status("Cleaning Bazel build artifacts...", "INFO")
        try:
            start_time = time.time()
            subprocess.run(
                bazel_prefix() + ["clean", "--expunge"], check=True, timeout=300
            )
            elapsed = time.time() - start_time
            self.timing_data["clean"] = elapsed
            print_status(f"Clean completed successfully ({elapsed:.2f}s)", "SUCCESS")
        except subprocess.CalledProcessError as e:
            print_status(f"Clean failed with exit code {e.returncode}", "ERROR")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status("Clean operation timed out (exceeded 5 minutes)", "ERROR")
            sys.exit(1)

    def build(self) -> None:
        if not self.run_build:
            return
        print_status("Starting Bazel build...", "INFO")
        self._shutdown_bazel_for_batch()
        cmd = self.build_bazel_command("build")
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["build"] = elapsed
            print_status(f"Build completed successfully ({elapsed:.2f}s)", "SUCCESS")
            if elapsed > self.slow_phase_warn_seconds:
                print_status(
                    f"Build took longer than {self.slow_phase_warn_seconds:.0f}s.",
                    "WARNING",
                )
        except subprocess.CalledProcessError as e:
            print_status(f"Build failed with exit code {e.returncode}", "ERROR")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(
                f"Build timed out (exceeded {self.subprocess_timeout}s)", "ERROR"
            )
            sys.exit(1)

    def test(self) -> None:
        if not self.run_tests:
            return
        print_status("Running Bazel tests...", "INFO")
        self._shutdown_bazel_for_batch()
        test_flags = [
            "--test_output=all" if self.verbose_tests else "--test_output=errors"
        ]
        if self.verbose_tests:
            test_flags.append("--verbose_failures")
        cmd = self.build_bazel_command("test", extra_flags=test_flags)
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=False, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["test"] = elapsed
            if result.returncode == 0:
                print_status(
                    f"Tests completed successfully ({elapsed:.2f}s)", "SUCCESS"
                )
            elif result.returncode == 4:
                print_status("No test targets found", "ERROR")
                sys.exit(4)
            else:
                print_status(
                    f"Tests failed with exit code {result.returncode}", "ERROR"
                )
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(
                f"Tests timed out (exceeded {self.subprocess_timeout}s)", "ERROR"
            )
            sys.exit(1)

    def coverage(self) -> None:
        if not self.run_coverage:
            return
        print_status("Running Bazel coverage...", "INFO")
        self._shutdown_bazel_for_batch()
        coverage_flags = [
            "--test_output=all" if self.verbose_tests else "--test_output=errors"
        ]
        cmd = self.build_bazel_command("coverage", extra_flags=coverage_flags)
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=False, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["coverage"] = elapsed
            if result.returncode == 0:
                print_status(
                    f"Coverage completed successfully ({elapsed:.2f}s)", "SUCCESS"
                )
                dat_path = self._build_llvm_coverage_report(start_time)
                if dat_path is None:
                    dat_path = self._resolve_coverage_dat_path()
                print_status(f"Coverage report (LCOV): {dat_path}", "INFO")
                self._generate_coverage_html(dat_path)
            elif result.returncode == 4:
                print_status("No test targets found", "ERROR")
                sys.exit(4)
            else:
                print_status(
                    f"Coverage failed with exit code {result.returncode}", "ERROR"
                )
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(
                f"Coverage timed out (exceeded {self.subprocess_timeout}s)", "ERROR"
            )
            sys.exit(1)

    def _bazel_info(self, key: str, fallback: str, match_config: bool = False) -> str:
        """Resolve a `bazel info <key>` value (absolute path, cwd-independent).

        `bazel-bin`/`output_path` are per-compilation_mode (fastbuild/dbg/opt)
        and per --repo_env (CC/CXX changes re-trigger cc_autoconf) — pass
        match_config=True so this resolves the *same* configuration as the
        build/test/coverage command that was just run, not a sibling one.
        """
        cmd = bazel_prefix() + ["info"]
        if match_config:
            cmd.extend(self._config_flags())
        cmd.append(key)
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=60
            )
            return result.stdout.strip()
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            return fallback

    def _resolve_coverage_dat_path(self) -> str:
        """Resolve Bazel's own (known-empty; see coverage() docstring) LCOV report."""
        output_path = self._bazel_info("output_path", "bazel-out", match_config=True)
        return os.path.join(output_path, "_coverage", "_coverage_report.dat")

    def _find_recent_profraw_files(self, since: float) -> list[str]:
        """Find .profraw files the just-run coverage tests produced.

        `bazel coverage` genuinely runs the instrumented tests and writes real
        .profraw data — it's only Bazel's own downstream report merging that's
        broken (its bundled collect_cc_coverage.sh never populates the
        runtime_objects_list.txt manifest entry llvm-cov needs). The raw files
        land under the output base's sandbox stash, which Bazel keeps around
        for sandbox reuse rather than deleting immediately.
        """
        output_base = self._bazel_info("output_base", "")
        if not output_base:
            return []
        stash_root = os.path.join(output_base, "sandbox", "sandbox_stash", "TestRunner")
        if not os.path.isdir(stash_root):
            return []
        profraw_files = []
        for root, _dirs, files in os.walk(stash_root):
            if "_coverage" not in root:
                continue
            for name in files:
                if not name.endswith(".profraw"):
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) >= since:
                        profraw_files.append(path)
                except OSError:
                    continue
        return profraw_files

    def _find_coverage_test_binaries(self) -> list[str]:
        """Resolve bazel-bin paths for every cc_test target covered by this run."""
        targets_expr = " + ".join(self.targets)
        try:
            query_result = subprocess.run(
                bazel_prefix()
                + [
                    "query",
                    f"kind(cc_test, {targets_expr})",
                    "--output=label",
                    # `query` doesn't inherit .bazelrc's "build --enable_workspace"
                    # the way build/test/coverage do (query only inherits "common",
                    # not "build"). This repo mixes bzlmod (MODULE.bazel: rules_cc,
                    # platforms, bazel_skylib) with WORKSPACE-registered
                    # local_repository/new_local_repository deps (fmt, magic_enum,
                    # glog, loguru, spdlog, googletest, benchmark) — querying
                    # without this flag fails to resolve those with "unknown repo"
                    # errors.
                    "--enable_workspace",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            return []

        bazel_bin = self._bazel_info("bazel-bin", "", match_config=True)
        if not bazel_bin:
            return []

        binaries = []
        for label in query_result.stdout.splitlines():
            label = label.strip()
            if not label.startswith("//"):
                continue
            pkg, _, name = label[2:].partition(":")
            binaries.append(os.path.join(bazel_bin, pkg, name))
        return binaries

    def _build_llvm_coverage_report(self, since: float) -> Optional[str]:
        """Build a real LCOV trace directly via llvm-profdata/llvm-cov (Clang only).

        Mirrors setup.py's CMake coverage dispatch: only the Clang path is
        implemented here, matching this repo's default toolchain.
        """
        if self.compiler != "clang":
            print_status(
                f"Bazel coverage HTML generation is only implemented for Clang "
                f"(compiler is '{self.compiler}'); falling back to Bazel's own "
                "(known-empty) report.",
                "WARNING",
            )
            return None

        fallback_suffix = "falling back to Bazel's own (known-empty) report"

        profraw_files = self._find_recent_profraw_files(since)
        if not profraw_files:
            print_status(
                "No .profraw files found from this coverage run; "
                f"{fallback_suffix}",
                "WARNING",
            )
            return None

        binaries = [
            b for b in self._find_coverage_test_binaries() if os.path.isfile(b)
        ]
        if not binaries:
            print_status(
                "Could not resolve test binaries for llvm-cov -object; "
                f"{fallback_suffix}",
                "WARNING",
            )
            return None

        llvm_profdata = shutil.which("llvm-profdata")
        llvm_cov = shutil.which("llvm-cov")
        if not llvm_profdata or not llvm_cov:
            print_status(
                f"llvm-profdata/llvm-cov not found on PATH; {fallback_suffix}",
                "WARNING",
            )
            return None

        output_path = self._bazel_info("output_path", "bazel-out", match_config=True)
        coverage_dir = os.path.join(output_path, "_coverage")
        os.makedirs(coverage_dir, exist_ok=True)
        merged_profdata = os.path.join(coverage_dir, "_merged.profdata")
        dat_path = os.path.join(coverage_dir, "_llvm_coverage_report.dat")

        try:
            merge_result = subprocess.run(
                [llvm_profdata, "merge", "-o", merged_profdata, *profraw_files],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
            if merge_result.returncode != 0:
                print_status(
                    f"llvm-profdata merge failed: {merge_result.stderr.strip()}",
                    "WARNING",
                )
                return None

            export_cmd = [
                llvm_cov,
                "export",
                "-instr-profile",
                merged_profdata,
                "-format=lcov",
                # /external/ is Bazel's staging path for non-vendored deps
                # (googletest, benchmark, ...) -- exclude those the same way
                # /ThirdParty/ (our vendored submodules) is excluded, so the
                # report only covers our own code.
                "-ignore-filename-regex="
                "(/external/|/ThirdParty/|^/opt/|^/usr/|^/Applications/)",
            ]
            for binary in binaries:
                export_cmd += ["-object", binary]
            export_result = subprocess.run(
                export_cmd,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
            if export_result.returncode != 0:
                print_status(
                    f"llvm-cov export failed: {export_result.stderr.strip()}", "WARNING"
                )
                return None

            # llvm-cov reports source paths as seen at compile time, i.e. under the
            # per-action sandbox (.../execroot/_main/<workspace-relative-path>) --
            # rewrite to the real, persistent repo path so genhtml can find the source.
            workspace_root = self._bazel_info("workspace", "")
            lcov_text = export_result.stdout
            if workspace_root:
                lcov_text = re.sub(
                    r"^SF:.*?/execroot/_main/",
                    f"SF:{workspace_root}/",
                    lcov_text,
                    flags=re.MULTILINE,
                )
            with open(dat_path, "w", encoding="utf-8") as f:
                f.write(lcov_text)
        except OSError as e:
            print_status(f"Failed to build coverage report: {e}", "WARNING")
            return None

        return dat_path

    def _generate_coverage_html(self, dat_path: str) -> None:
        """Convert the Bazel LCOV coverage report into an HTML report via genhtml."""
        if not os.path.isfile(dat_path):
            print_status(
                f"Coverage LCOV file not found at {dat_path}; skipping HTML report",
                "WARNING",
            )
            return
        genhtml = shutil.which("genhtml")
        if genhtml is None:
            print_status(
                "genhtml not found on PATH; skipping HTML report (install lcov, "
                "e.g. 'brew install lcov')",
                "WARNING",
            )
            return
        html_dir = os.path.join(os.path.dirname(dat_path), "html")
        cmd = [
            genhtml,
            dat_path,
            "--output-directory",
            html_dir,
            # llvm-cov's LCOV export and genhtml's stricter lcov-2.x consistency
            # checker don't always agree on function-vs-line hit counts for
            # lambdas/closures (e.g. inside googletest internals).
            "--ignore-errors",
            "inconsistent,corrupt,unsupported",
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=self.subprocess_timeout,
                capture_output=True,
                text=True,
            )
            index_html = os.path.abspath(os.path.join(html_dir, "index.html"))
            print_status(f"Coverage HTML report: {index_html}", "SUCCESS")
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "").strip()[-500:]
            print_status(
                f"Failed to generate HTML coverage report: {stderr_tail}",
                "WARNING",
            )
        except subprocess.TimeoutExpired as e:
            print_status(f"Failed to generate HTML coverage report: {e}", "WARNING")

    def print_timing_summary(self) -> None:
        if not self.timing_data:
            return
        print("\n" + "=" * 80)
        print("BUILD TIMING SUMMARY")
        print("=" * 80)
        total_time = sum(self.timing_data.values())
        for phase, elapsed in self.timing_data.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {phase.capitalize():20} {elapsed:8.2f}s ({percentage:5.1f}%)")
        print(f"  {'-' * 40}")
        print(f"  {'Total':20} {total_time:8.2f}s (100.0%)")
        print("=" * 80 + "\n")

    def execute(self) -> None:
        """Execute the build pipeline."""
        self.print_configuration_summary()
        self.config()
        if self.run_config and (self.run_build or self.run_tests or self.run_coverage):
            print_status(
                "Config requested: forcing clean build (bazel clean --expunge).", "INFO"
            )
            self.run_clean = True
        self.clean()
        if self.run_build or self.run_tests or self.run_coverage:
            self._kill_stale_bazel_processes()
        # `bazel coverage` requires instrumented compilation (different flags than a
        # plain build/test) and already builds + tests everything itself, so a
        # preceding plain build/test would just be wasted, non-reusable work.
        if self.run_coverage:
            if self.run_build or self.run_tests:
                print_status(
                    "Coverage requested: skipping separate build/test steps "
                    "(bazel coverage builds and tests everything itself under "
                    "different instrumented flags).",
                    "INFO",
                )
            self.coverage()
        else:
            self.build()
            self.test()
        self.print_timing_summary()


def parse_args(args: list[str]) -> list[str]:
    """Parse argv like Scripts/setup.py: long flags, dotted shortcuts."""
    processed: list[str] = []
    for arg in args:
        if arg.startswith("--backend."):
            bt = arg.split(".", 1)[1].lower()
            if bt in _BACKEND_NAMES:
                processed.append(bt)
            else:
                print_status(
                    f"Invalid logging backend: {bt}. Valid: native, loguru, "
                    "glog, spdlog",
                    "ERROR",
                )
                sys.exit(1)
            continue
        if arg.startswith("--sanitizer."):
            processed.append(f"sanitizer.{arg.split('.', 1)[1].lower()}")
            continue
        if arg.startswith("--lto."):
            processed.append(f"lto.{arg.split('.', 1)[1].lower()}")
            continue
        if arg.startswith("--linker."):
            processed.append(f"linker.{arg.split('.', 1)[1].lower()}")
            continue
        if re.search(r"[/\\]", arg) and re.search(
            r"[Cc]lang|[Gg][Cc][Cc]|[Gg]\+\+", arg
        ):
            processed.append(arg)
            continue
        if "." in arg and not arg.startswith("--"):
            processed.extend(_merge_dotted_segments(arg.split(".")))
            continue
        processed.append(arg.lower())
    return processed


def print_help() -> None:
    print_status("LinearAlgebra Bazel Build Configuration Helper", "INFO")
    print("\n" + "=" * 80)
    print("BAZEL BUILD SYSTEM")
    print("=" * 80)
    print("\nUsage examples:")
    print("  1. Show configuration (no build):")
    print("     python setup_bazel.py config")
    print("  2. Default debug build:")
    print("     python setup_bazel.py build.test")
    print("  3. Release build:")
    print("     python setup_bazel.py build.test.release")
    print("  4. Build against a specific logging backend:")
    print("     python setup_bazel.py build.test.glog")
    print("     python setup_bazel.py build.test --backend.spdlog")
    print("  5. Run tests only:")
    print("     python setup_bazel.py test")
    print("  6. Coverage (llvm-profdata/llvm-cov + genhtml report; Clang only):")
    print("     python setup_bazel.py coverage")
    print("  7. Clean build:")
    print("     python setup_bazel.py clean.build.test.release")
    print("  8. MKL + AVX2, CUDA with explicit SM list:")
    print("     python setup_bazel.py build.test.mkl.avx2")
    print("     python setup_bazel.py build.cuda.cudaarch.sm_80,sm_86")
    print("  9. ccache + ThinLTO:")
    print("     python setup_bazel.py build.cache.ccache.lto.thin")
    print("\nBuild types:")
    print("  debug          - Debug build, -c dbg (default)")
    print("  release        - Release build, -c opt")
    print("  relwithdebinfo - -c opt --copt=-g")
    print("  minsizerel     - -c opt --copt=-Os (/O1 for MSVC)")
    print("\nCompiler (via --repo_env=CC=/CXX=; default: Clang):")
    print("  clang | gcc | clang-15 | gcc-13 | ...")
    print("\nC++ Standard:")
    print("  cxx20 (default, from .bazelrc) | cxx23")
    print("\nLogging backend (mirrors CMake LOGGING_BACKEND; default loguru):")
    print("  native | loguru | glog | spdlog")
    print(
        "  --backend.native | --backend.loguru | --backend.glog | "
        "--backend.spdlog  (same, long form)"
    )
    print("\nLogging feature toggles (mirror CMake LOGGING_ENABLE_*):")
    print(
        "  nomagicenum    - --define=logging_enable_magic_enum=false "
        "(default: enabled)"
    )
    print(
        "  nocxademangle  - --define=logging_enable_cxa_demangle=false "
        "(default: enabled, non-Windows)"
    )
    print(
        "  portablefloat  - --define=logging_enable_portable_float_format=true "
        "(default: off)"
    )
    print("  stdformat      - --define=logging_format_use_std=true (default: off)")
    print(
        "  logfatal       - --define=logging_default_exception_mode=log_fatal "
        "(default: throw)"
    )
    print("\nLinking:")
    print("  shared         - --define=build_shared_libs=true")
    print("  static         - Bazel's own default (no define needed)")
    print("\nLinearAlgebra numerics (mirror CMake LINALG_ENABLE_*):")
    print("  mkl            - --define=linalg_enable_mkl=true")
    print("  blas           - --define=linalg_enable_blas=true")
    print("  blis           - same as 'blas' (Bazel has no separate BLIS library")
    print("                   resolution; see module docstring)")
    print("  lupivoting     - --define=linalg_lu_pivoting=true")
    print("  win32warnings  - --define=linalg_display_win32_warnings=true")
    print("\nGPU backend (mirrors CMake LINALG_GPU_BACKEND; default none):")
    print("  cuda | hip     - --define=linalg_gpu_backend=cuda|hip")
    print("  gpu.cuda | gpu.hip | gpu.none  (same, long form)")
    print("  cudaarch.sm_80,sm_86   - --@rules_cuda//cuda:archs=sm_80,sm_86")
    print(
        "  hiparch.gfx90a,gfx1100 - --copt=--offload-arch=gfx90a "
        "--copt=--offload-arch=gfx1100"
    )
    print("\nCPU SIMD backend (mirrors CMake LINALG_CPU_BACKEND; default no):")
    print("  no | sse | avx | avx2 | avx512 | neon | sve")
    print("  cpu.avx2  (same, long form)")
    print("\nProfiling (mirrors CMake LINALG_ENABLE_PROFILING):")
    print("  profiling      - --define=linalg_enable_profiling=true; requires an")
    print("                   installed KhwarizmiAnalytix/Profiler and")
    print("                   --repo_env=PROFILER_PATH=/its/install/prefix")
    print("                   (bazel/profiler_repository.bzl)")
    print(
        "\nSanitizers (--copt/--linkopt=-fsanitize=...; "
        "mirrors CMake LINALG_SANITIZER_TYPE):"
    )
    print("  sanitizer.address | sanitizer.undefined | sanitizer.thread |")
    print("  sanitizer.memory (Clang only) | sanitizer.leak (not Windows/macOS) |")
    print("  sanitizer.address,undefined   (comma-combine)")
    print("\nLTO (--copt/--linkopt=-flto[=thin]; mirrors CMake LINALG_LTO_MODE):")
    print("  lto.off | lto.thin (Clang) | lto.full | lto.ipo | lto.auto")
    print(
        "  (skipped automatically under coverage/sanitizer builds, "
        "like Cmake/lto.cmake)"
    )
    print("\nLinker (--linkopt=-fuse-ld=...; mirrors CMake LINALG_LINKER_CHOICE):")
    print("  linker.default | linker.lld | linker.mold | linker.gold")
    print(
        "\nCompiler cache (generates a launcher script; "
        "mirrors CMake LINALG_CACHE_BACKEND):"
    )
    print("  cache.ccache | cache.sccache | cache.buildcache   (non-Windows only)")
    print("  cache          - no backend named; matches CMake's none-backend no-op")
    print(
        "\nDistributed compilation (mirrors CMake LINALG_ENABLE_ICECC; "
        "non-Windows only):"
    )
    print("  icecc          - chains with cache.* if both are given")
    print("\nMisc:")
    print("  vv             - Verbose Bazel test output (--test_output=all)")
    print("  batch          - Pass --batch to Bazel (runs `bazel shutdown` first)")
    print("\nActions:")
    print("  config         - Show configuration summary (no build)")
    print("  build          - Build the project")
    print("  test           - Run tests")
    print("  coverage       - Run tests with coverage instrumentation (lcov report)")
    print("  clean          - Clean build artifacts")
    print("\nCMake-only (Scripts/setup.py), not wired to Bazel in this repo:")
    print(
        "  metal | gpu.metal | examples | cppcheck | clangtidy | fix | iwyu | "
        "spell |"
    )
    print("  valgrind | usesystemlogging | usesystemscalars | enableexternal")
    print("\nFlag parity with .github/scripts/flag_matrix.py:")
    print("  See this script's module docstring for the full three-bucket breakdown")
    print(
        "  (source-gating --define=, raw compiler/linker flags, "
        "genuinely CMake-only)."
    )
    print("\nEquivalent to CMake setup.py:")
    print("  CMake:  python setup.py config.build.test.release")
    print("  Bazel:  python setup_bazel.py config.build.test.release")
    print()


def main() -> None:
    from helpers.cli import dispatch_bazel

    forwarded = dispatch_bazel(sys.argv[1:])
    if forwarded is not None:
        sys.exit(forwarded)

    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print_help()
        return

    if len(sys.argv) < 2:
        print_status(
            "No build configuration specified. Use --help for usage information.",
            "ERROR",
        )
        sys.exit(1)

    try:
        arg_list = parse_args(sys.argv[1:])
        print_status(f"Starting Bazel build for {platform.system()}", "INFO")
        config = LinearAlgebraBazelConfiguration(arg_list)

        if not (
            config.run_build
            or config.run_tests
            or config.run_clean
            or config.run_config
            or config.run_coverage
        ):
            config.run_build = True

        config.execute()
        print_status("Build process completed successfully!", "SUCCESS")

    except KeyboardInterrupt:
        print_status("\nBuild process interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
