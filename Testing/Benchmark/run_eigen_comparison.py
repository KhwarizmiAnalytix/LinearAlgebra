"""Run every registered Eigen/library benchmark sequentially and preserve raw results."""

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import shutil
import json
import os
import re
from pathlib import Path
import subprocess
import time

OPERATIONS = (
    "multiplication",
    "ludecomposition",
    "schurdecomposition",
    "qrdecomposition",
    "choleskydecomposition",
    "svddecomposition",
    "eigen",
    "kernels",
)


def capture(command, cwd=None):
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def run(build, output, timeout, operations=OPERATIONS, pattern=""):
    """Run selected cases after checking the build and its unit tests."""
    selected = re.compile(pattern)
    root = Path(__file__).resolve().parents[2]
    build, output = build.resolve(), output.resolve()
    if (output / "runs.json").exists():
        raise RuntimeError("Output already contains a run; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    env = dict(os.environ)
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[variable] = "1"
    tested = subprocess.run(
        ["ctest", "--test-dir", str(build), "-L", "Unit", "--output-on-failure"],
        env=env,
        text=True,
        capture_output=True,
    )
    (output / "tests.log").write_text(tested.stdout + tested.stderr)
    detail = build / "Testing/Temporary/LastTest.log"
    if detail.exists():
        with (
            detail.open("rb") as source,
            gzip.open(output / "tests-detail.log.gz", "wb") as target,
        ):
            shutil.copyfileobj(source, target)
    if tested.returncode:
        raise RuntimeError("Unit tests failed; see tests.log")
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if cache.get("CMAKE_BUILD_TYPE", "").lower() != "release":
        raise RuntimeError("Use a Release build for benchmark results")
    if cache.get("LINALG_ENABLE_BENCHMARK") != "ON":
        raise RuntimeError("Configure with LINALG_ENABLE_BENCHMARK=ON")
    hardware = (
        capture(["system_profiler", "SPHardwareDataType"])
        if os.uname().sysname == "Darwin"
        else ""
    )
    hardware = [
        line.strip()
        for line in hardware.splitlines()
        if line.strip().startswith(
            ("Chip:", "Memory:", "Total Number of Cores:", "Model Name:")
        )
    ]
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "build_directory": str(build),
        "hardware": hardware,
        "os": capture(["uname", "-smr"]),
        "compiler": capture([cache["CMAKE_CXX_COMPILER"], "--version"]),
        "git_commit": capture(["git", "rev-parse", "HEAD"], root),
        "working_tree_status": capture(["git", "status", "--short"], root),
        "eigen_commit": capture(
            ["git", "-C", str(root / "ThirdParty/eigen"), "rev-parse", "HEAD"]
        ),
        "cache": {
            k: v
            for k, v in cache.items()
            if k.startswith(("LINALG_", "BLAS_", "LAPACK_"))
            or k in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE")
        },
        "thread_environment": {
            key: env[key]
            for key in env
            if key.endswith("NUM_THREADS") or key == "VECLIB_MAXIMUM_THREADS"
        },
        "timeout_per_case_seconds": timeout,
        "operations": list(operations),
        "filter": pattern,
        "library_environment": {
            key: env[key]
            for key in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")
            if key in env
        },
        "method": "Sequential cases; registered minimum 0.1 seconds and three repetitions; medians of CPU time.",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                list((root / "Testing/Benchmark").glob("Benchmark*.cxx"))
                + list((root / "src").rglob("*.cpp"))
                + list((root / "src").rglob("*.h"))
                + [root / "CMakeLists.txt"]
            )
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    index = []
    for operation in operations:
        executable = build / "bin" / f"benchmark_linalg_{operation}"
        names = capture([str(executable), "--benchmark_list_tests=true"]).splitlines()
        names = [name for name in names if selected.search(name)]
        print(f"{operation}: {len(names)} selected cases", flush=True)
        for number, name in enumerate(names, 1):
            stem = f"{operation}-{number:03}"
            destination = raw / f"{stem}.json"
            command = [
                str(executable),
                "--benchmark_filter=^" + name.replace(".", "[.]") + "$",
                "--benchmark_out=" + str(destination),
                "--benchmark_out_format=json",
            ]
            item = {
                "operation": operation,
                "name": name,
                "command": command,
                "json": f"raw/{stem}.json",
                "log": f"raw/{stem}.log",
            }
            started = time.monotonic()
            try:
                process = subprocess.run(
                    command, env=env, text=True, capture_output=True, timeout=timeout
                )
                (raw / f"{stem}.log").write_text(process.stdout + process.stderr)
                item["exit_code"] = process.returncode
                item["status"] = "ok" if process.returncode == 0 else "failed"
                if process.returncode == 0:
                    try:
                        records = json.loads(destination.read_text())["benchmarks"]
                    except (OSError, ValueError, KeyError) as error:
                        records = []
                        item["errors"] = [str(error)]
                    if not records or any(row.get("error_occurred") for row in records):
                        item["status"] = "failed"
                        item["errors"] = [
                            row.get("error_message", "Empty benchmark result")
                            for row in records
                            if row.get("error_occurred")
                        ]
            except subprocess.TimeoutExpired as error:
                stdout, stderr = error.stdout or b"", error.stderr or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                (raw / f"{stem}.log").write_text(stdout + stderr)
                item["status"] = "timeout"
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
            index.append(item)
            (output / "runs.json").write_text(json.dumps(index, indent=2) + "\n")
            print(
                f"  {number:3}/{len(names)} {item['status']:7} {item['elapsed_seconds']:6.2f}s {name}",
                flush=True,
            )
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["successful_cases"] = sum(item["status"] == "ok" for item in index)
    metadata["total_cases"] = len(index)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not index:
        raise RuntimeError("No benchmarks matched the selection")
    print(
        f"Completed: {metadata['successful_cases']}/{len(index)} cases passed",
        flush=True,
    )

    return 0 if metadata["successful_cases"] == len(index) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--operations", nargs="+", choices=OPERATIONS, default=OPERATIONS
    )
    parser.add_argument(
        "--filter", default="", help="Python regular expression matching case names"
    )
    args = parser.parse_args()
    raise SystemExit(
        run(args.build_dir, args.output_dir, args.timeout, args.operations, args.filter)
    )
