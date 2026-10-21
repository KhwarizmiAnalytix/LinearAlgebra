"""Build each selected flag assignment and verify effective configuration and tests."""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from flag_matrix import PLATFORMS


def command(args, log, env):
    """Keep command failures and their complete output in per-case artifacts."""
    with log.open("a", encoding="utf-8") as output:
        output.write("\n" + repr([str(arg) for arg in args]) + "\n")
        output.flush()
        subprocess.run(
            [str(arg) for arg in args],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def runtime_skip(flags):
    """Report builds that cannot be executed safely on an ordinary hosted runner."""
    if flags["LINALG_ENABLE_SANITIZER"] == "ON":
        if flags["LINALG_SANITIZER_TYPE"] == "memory":
            return (
                "MemorySanitizer requires an instrumented C++ runtime and dependencies"
            )
        if flags["LINALG_SANITIZER_TYPE"] == "thread":
            return (
                "ThreadSanitizer runtime requires a compatible virtual address layout"
            )
    if flags["LINALG_ENABLE_VECTORIZATION"] == "ON":
        isa = flags["LINALG_VECTORIZATION_ISA"]
        required = {
            "avx": {"avx"},
            "avx2": {"avx2", "fma"},
            "avx512": {"avx512f", "avx2", "fma"},
        }.get(isa, set())
        if required:
            cpuinfo = Path("/proc/cpuinfo")
            available = set(cpuinfo.read_text().split()) if cpuinfo.exists() else set()
            if not required <= available:
                return f"Runner ISA capabilities do not establish {isa} support"
    return None


def installed_logging(source, work, shared, config, env, log):
    """Produce a real reusable Logging package through the project's integration boundary."""
    package = f"logging-{shared}-{config}"
    build, prefix = work / package, work / f"{package}-prefix"
    if not (prefix / "lib/cmake/Logging/LoggingConfig.cmake").is_file():
        command(
            [
                "cmake",
                "-S",
                source / ".github/cmake/logging-package",
                "-B",
                build,
                "-G",
                "Ninja",
                f"-DLINALG_SOURCE_DIR={source}",
                f"-DBUILD_SHARED_LIBS={shared}",
                f"-DCMAKE_BUILD_TYPE={config}",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
            ],
            log,
            env,
        )
        command(
            [
                "cmake",
                "--build",
                build,
                "--config",
                config,
                "--target",
                "Logging",
                "--parallel",
                "4",
            ],
            log,
            env,
        )
        command(["cmake", "--install", build / "logging", "--config", config], log, env)
    return prefix


def verify(build, flags):
    """Check the realized target contract, including options that may otherwise be ignored."""
    state = dict(
        line.split("=", 1)
        for line in (build / "ci-effective-flags.txt").read_text().splitlines()
    )
    testing = flags["BUILD_TESTING"] == flags["LINALG_ENABLE_TESTING"] == "ON"
    assert (state["LINALG_ENABLE_TESTING"] == "ON") == testing, state
    assert state["TYPE"] == (
        "SHARED_LIBRARY" if flags["BUILD_SHARED_LIBS"] == "ON" else "STATIC_LIBRARY"
    ), state
    assert state["CXX_STANDARD"] == flags["LINALG_CXX_STANDARD"], state
    assert state["benchmark_linalg_eigen"] == flags["LINALG_ENABLE_BENCHMARK"], state
    assert state["LINALG_ENABLE_BLIS"] == flags["LINALG_ENABLE_BLIS"], state
    assert state["LINALG_ENABLE_BLAS"] == (
        "ON" if flags["LINALG_ENABLE_BLIS"] == "ON" else flags["LINALG_ENABLE_BLAS"]
    ), state
    instrumentation = (
        flags["LINALG_ENABLE_COVERAGE"] == "ON"
        or flags["LINALG_ENABLE_SANITIZER"] == "ON"
    )
    if instrumentation or flags["LINALG_LTO_MODE"] == "off":
        assert state["LINALG_EFFECTIVE_LTO"] == "off", state
    else:
        assert state["LINALG_EFFECTIVE_LTO"] != "off", state
    cache = (
        flags["LINALG_ENABLE_CACHE"] == "ON" and flags["LINALG_CACHE_BACKEND"] != "none"
    )
    if cache:
        assert state["LINALG_CACHE_BACKEND"] == flags["LINALG_CACHE_BACKEND"], state
        assert (
            flags["LINALG_CACHE_BACKEND"] in state["CXX_COMPILER_LAUNCHER"].lower()
        ), state
    if flags["LINALG_ENABLE_ICECC"] == "ON":
        assert "icecc" in state["CXX_COMPILER_LAUNCHER"].lower(), state
    if flags["LINALG_ENABLE_SANITIZER"] == "ON":
        expected = "-fsanitize=" + flags["LINALG_SANITIZER_TYPE"]
        assert (
            expected in state["COMPILE_OPTIONS"] and expected in state["LINK_OPTIONS"]
        ), state
    if flags["LINALG_ENABLE_COVERAGE"] == "ON":
        assert any(
            marker in state["COMPILE_OPTIONS"]
            for marker in ("--coverage", "-fprofile-instr-generate", "/Zi", "/Z7")
        ), state
    if flags["LINALG_ENABLE_VECTORIZATION"] == "ON":
        assert (
            "LINALG_SIMD_" + flags["LINALG_VECTORIZATION_ISA"].upper()
            in state["COMPILE_DEFINITIONS"]
        ), state
    for option, definition in (
        ("LINALG_LU_PIVOTING", "LINALG_HAS_LU_PIVOTING"),
        ("LINALG_DISPLAY_WIN32_WARNINGS", "LINALG_HAS_WIN32_WARNINGS"),
        ("LINALG_ENABLE_VECTORIZATION", "LINALG_HAS_VECTORIZATION"),
    ):
        assert (
            f"{definition}=$<BOOL:{flags[option]}>" in state["COMPILE_DEFINITIONS"]
        ), state
    if flags["LINALG_LINKER_CHOICE"] != "default":
        assert flags["LINALG_LINKER_CHOICE"] in state["LINK_OPTIONS"], state
    return testing


def run_case(case_id, case, source, work):
    flags = case["flags"]
    build, log = work / case_id, work / "logs" / f"{case_id}.log"
    env = os.environ.copy()
    env.update(
        CC=PLATFORMS[case["platform"]]["cc"],
        CXX=PLATFORMS[case["platform"]]["cxx"],
        ICECC="no",
    )
    env.update(
        CCACHE_DIR=str(work / "ccache"),
        SCCACHE_DIR=str(work / "sccache"),
        BUILDCACHE_DIR=str(work / "buildcache"),
    )
    # Logging packages, optional BLAS DLLs, and Clang's ASan runtime must be on PATH on Windows.
    prefix = os.environ.get("CMAKE_PREFIX_PATH", "")
    if flags["LINALG_USE_SYSTEM_LOGGING"] == "ON":
        logging = installed_logging(
            source,
            work,
            flags["BUILD_SHARED_LIBS"],
            flags["CMAKE_BUILD_TYPE"],
            env,
            log,
        )
        prefix = str(logging) + (";" + prefix if prefix else "")
        env["PATH"] = str(logging / "bin") + os.pathsep + env["PATH"]
    options = [f"-D{key}={value}" for key, value in flags.items() if key != "generator"]
    if flags["generator"] == "Ninja Multi-Config":
        # Ninja's defaults omit MinSizeRel, which is part of the flag domain.
        options.append(
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo;MinSizeRel"
        )
    options.extend(
        [
            "-DCMAKE_DISABLE_FIND_PACKAGE_cpuinfo=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Profiler=ON",
            "-DLINALG_ENABLE_MKL=OFF",
            "-DLINALG_ENABLE_CUBLAS=OFF",
            "-DLINALG_BUILD_EXAMPLES=OFF",
            "-DLINALG_ENABLE_PROFILING=OFF",
            "-DLINALG_ENABLE_VALGRIND=OFF",
            "-DLINALG_ENABLE_CLANGTIDY=OFF",
            "-DLINALG_ENABLE_FIX=OFF",
            "-DLINALG_ENABLE_IWYU=OFF",
            "-DLINALG_ENABLE_SPELL=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_PROJECT_LinearAlgebra_INCLUDE={source / '.github/cmake/verify-flags.cmake'}",
        ]
    )
    if prefix:
        options.append(f"-DCMAKE_PREFIX_PATH={prefix}")
    if flags["LINALG_ENABLE_BLAS"] == "ON" and flags["LINALG_ENABLE_BLIS"] == "OFF":
        options.append("-DBLA_VENDOR=OpenBLAS")
    if case["platform"] == "linux-clang":
        options.extend(
            [
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]
        )
    command(
        ["cmake", "-S", source, "-B", build, "-G", flags["generator"], *options],
        log,
        env,
    )
    testing = verify(build, flags)
    command(
        [
            "cmake",
            "--build",
            build,
            "--config",
            flags["CMAKE_BUILD_TYPE"],
            "--parallel",
            "4",
            "--verbose",
        ],
        log,
        env,
    )
    listing = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build),
            "-C",
            flags["CMAKE_BUILD_TYPE"],
            "--show-only=json-v1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    names = {test["name"] for test in json.loads(listing.stdout)["tests"]}
    assert ("GpuResidencySourceCheck" in names) == testing, names
    assert ("LinearAlgebraCxxTests" in names) == (
        testing and flags["LINALG_ENABLE_GTEST"] == "ON"
    ), names
    skip = runtime_skip(flags)
    if testing:
        args = [
            "ctest",
            "--test-dir",
            build,
            "-C",
            flags["CMAKE_BUILD_TYPE"],
            "--output-on-failure",
            "--no-tests=error",
            "-LE",
            "Performance|CUDA",
        ]
        if skip:
            args.extend(["-R", "^GpuResidencySourceCheck$"])
        command(args, log, env)
    return {
        "status": "passed",
        "runtime": "compile-only: " + skip
        if skip
        else (
            "CPU tests"
            if testing and flags["LINALG_ENABLE_GTEST"] == "ON"
            else (
                "source check only (GoogleTest disabled)"
                if testing
                else "testing disabled by flags"
            )
        ),
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    shard = next(
        job for job in manifest["matrix"]["include"] if job["shard"] == args.shard
    )
    spec = PLATFORMS[shard["platform"]]
    if (spec["arch"] == "arm") != (platform.machine().lower() in ("arm64", "aarch64")):
        raise RuntimeError(
            "Runner architecture does not match the generated ISA domains"
        )
    work = args.work.resolve()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    results = {}
    for case_id in shard["case_ids"]:
        try:
            results[case_id] = run_case(
                case_id,
                manifest["cases"][case_id],
                Path(__file__).resolve().parents[2],
                work,
            )
        except (subprocess.CalledProcessError, AssertionError, OSError) as error:
            results[case_id] = {
                "status": "failed",
                "error": str(error),
                "flags": manifest["cases"][case_id]["flags"],
            }
        print(case_id + ": " + results[case_id]["status"], flush=True)
        (work / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = [key for key, result in results.items() if result["status"] == "failed"]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write("| Case | Result | Runtime scope |\n|---|---|---|\n")
            for key, result in results.items():
                output.write(
                    f"| {key} | {result['status']} | {result.get('runtime', 'see artifact log')} |\n"
                )
    if failed:
        raise SystemExit("Flag combinations failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
