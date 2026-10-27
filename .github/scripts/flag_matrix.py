"""Generate constrained pairwise builds or windows of the exhaustive flag space."""

import argparse
import itertools
import json
import math
import random
import re
from pathlib import Path

BOOL = ("OFF", "ON")
PLATFORMS = {
    "linux-gcc": {
        "os": "ubuntu-24.04",
        "cc": "gcc",
        "cxx": "g++",
        "kind": "gcc",
        "arch": "x86",
    },
    "linux-clang": {
        "os": "ubuntu-24.04",
        "cc": "clang",
        "cxx": "clang++",
        "kind": "clang",
        "arch": "x86",
    },
    "macos-clang": {
        "os": "macos-15",
        "cc": "clang",
        "cxx": "clang++",
        "kind": "clang",
        "arch": "arm",
    },
    "windows-msvc": {
        "os": "windows-2022",
        "cc": "cl",
        "cxx": "cl",
        "kind": "msvc",
        "arch": "x86",
    },
    "windows-clang": {
        "os": "windows-2022",
        "cc": "clang",
        "cxx": "clang++",
        "kind": "clang",
        "arch": "x86",
    },
}
# These options require dependency/hardware or developer-tool contracts rather
# than ordinary CPU builds. Inventory validation prevents new switches silently
# falling outside either the pairwise matrix or an explicit regression suite.
FOCUSED = {
    "LINALG_ENABLE_MKL": "ci.yml: cmake-mkl and cmake-mkl-windows (native and combined BLAS)",
    "LINALG_GPU_BACKEND": "ci.yml: cuda-compile, cuda-windows, hip-compile, and metal-compile; GPU runtime remains gated",
    "LINALG_BUILD_EXAMPLES": "ci.yml: CUDA/HIP + MKL + OpenBLAS example compilation",
    "LINALG_ENABLE_PROFILING": "check_tool_flags.py: available/missing Profiler target contracts",
    "LINALG_ENABLE_CLANGTIDY": "check_tool_flags.py: all developer-tool switch combinations",
    "LINALG_ENABLE_FIX": "check_tool_flags.py: recorded fix arguments, without rewriting sources",
    "LINALG_ENABLE_IWYU": "check_tool_flags.py: all developer-tool switch combinations",
    "LINALG_ENABLE_SPELL": "check_tool_flags.py: all developer-tool switch combinations",
    "LINALG_ENABLE_VALGRIND": "ci.yml: Linux memcheck; dashboard/timeout contracts on all platforms",
}


def domains(platform):
    """Return meaningful values available on one hosted toolchain."""
    spec = PLATFORMS[platform]
    windows = platform.startswith("windows")
    macos = platform.startswith("macos")
    sanitizers = ("address", "undefined", "address,undefined")
    if not windows and not macos:
        sanitizers += ("thread", "leak")
        if spec["kind"] == "clang":
            sanitizers += ("memory",)
    elif windows:
        sanitizers = ("address",)
    lto = ("off", "full", "ipo", "auto")
    if spec["kind"] == "clang":
        lto += ("thin",)
    return {
        "BUILD_SHARED_LIBS": BOOL,
        "CMAKE_BUILD_TYPE": ("Release", "Debug", "RelWithDebInfo", "MinSizeRel"),
        "LINALG_CXX_STANDARD": ("20", "23"),
        "generator": ("Ninja", "Ninja Multi-Config"),
        "BUILD_TESTING": BOOL,
        "LINALG_ENABLE_TESTING": BOOL,
        "LINALG_ENABLE_GTEST": BOOL,
        "LINALG_ENABLE_BLAS": BOOL,
        "LINALG_ENABLE_BLIS": ("OFF",) if windows else BOOL,
        "LINALG_CPU_BACKEND": ("no", "neon", "sve")
        if spec["arch"] == "arm"
        else ("no", "sse", "avx", "avx2", "avx512"),
        "LINALG_LU_PIVOTING": BOOL,
        "LINALG_DISPLAY_WIN32_WARNINGS": BOOL,
        "LINALG_ENABLE_CACHE": BOOL,
        "LINALG_CACHE_BACKEND": ("none", "ccache", "sccache", "buildcache"),
        "LINALG_ENABLE_ICECC": ("OFF",) if windows else BOOL,
        "LINALG_LTO_MODE": lto,
        "LINALG_LINKER_CHOICE": ("default", "lld", "mold", "gold")
        if platform.startswith("linux")
        else ("default",),
        "LINALG_ENABLE_COVERAGE": BOOL,
        "LINALG_ENABLE_SANITIZER": ("OFF",) if spec["kind"] == "msvc" else BOOL,
        "LINALG_SANITIZER_TYPE": sanitizers,
        "LINALG_ENABLE_BENCHMARK": BOOL,
        "LINALG_ENABLE_EXTERNAL": BOOL,
        "LINALG_USE_SYSTEM_LOGGING": BOOL,
        "LINALG_USE_SYSTEM_SCALARS": BOOL,
    }


def valid(partial, platform=None):
    """Reject only combinations with an explicit unsupported build contract."""
    active_cache = partial.get("LINALG_ENABLE_CACHE") == "ON"
    backend = partial.get("LINALG_CACHE_BACKEND")
    if (
        active_cache
        and backend == "buildcache"
        and partial.get("LINALG_ENABLE_ICECC") == "ON"
    ):
        return False
    if (
        active_cache
        and backend == "sccache"
        and partial.get("LINALG_LTO_MODE", "off") != "off"
        and partial.get("LINALG_ENABLE_COVERAGE") == "OFF"
        and partial.get("LINALG_ENABLE_SANITIZER") == "OFF"
    ):
        return False
    if (
        platform == "linux-gcc"
        and partial.get("LINALG_LINKER_CHOICE") == "lld"
        and partial.get("LINALG_LTO_MODE", "off") != "off"
        and partial.get("LINALG_ENABLE_COVERAGE") == "OFF"
        and partial.get("LINALG_ENABLE_SANITIZER") == "OFF"
    ):
        # LLVM's linker cannot consume GCC LTO objects.
        return False
    return True


def complete(factors, partial, rng=None, platform=None):
    """Find a valid full assignment extending a partial assignment."""
    if not valid(partial, platform):
        return None
    missing = [key for key in factors if key not in partial]
    if not missing:
        return dict(partial)
    key = missing[0]
    values = list(factors[key])
    if rng:
        rng.shuffle(values)
    for value in values:
        result = complete(factors, {**partial, key: value}, rng, platform)
        if result is not None:
            return result
    return None


def pairs(row):
    """Represent each pair independently of dictionary insertion order."""
    return {
        ((a, row[a]), (b, row[b])) for a, b in itertools.combinations(sorted(row), 2)
    }


def pairwise(factors, platform=None):
    """Cover every extendable value pair, retaining feasibility witnesses."""
    candidates = {}
    feasible = set()
    keys = sorted(factors)
    for a, b in itertools.combinations(keys, 2):
        for av, bv in itertools.product(factors[a], factors[b]):
            witness = complete(factors, {a: av, b: bv}, platform=platform)
            if witness is not None:
                feasible.add(((a, av), (b, bv)))
                candidates[tuple(witness[key] for key in factors)] = witness
    rng = random.Random(20261021)
    for _ in range(256):
        witness = complete(factors, {}, rng, platform)
        candidates[tuple(witness[key] for key in factors)] = witness
    pool = [(row, pairs(row)) for row in candidates.values()]
    uncovered = set(feasible)
    selected = []
    while uncovered:
        index = max(range(len(pool)), key=lambda i: len(pool[i][1] & uncovered))
        row, covered = pool.pop(index)
        if not covered & uncovered:
            raise AssertionError("No candidate covers an outstanding feasible pair")
        selected.append(row)
        uncovered.difference_update(covered)
    return selected, len(feasible)


def exhaustive_window(factors, offset, limit, platform=None):
    """Decode a bounded Cartesian window, including invalid slots in the ledger."""
    size = math.prod(map(len, factors.values()))
    if offset >= size:
        raise ValueError(f"Offset {offset} is outside the {size} Cartesian assignments")
    rows = []
    stop = min(size, offset + limit)
    for index in range(offset, stop):
        quotient = index
        row = {}
        for key, values in reversed(list(factors.items())):
            quotient, digit = divmod(quotient, len(values))
            row[key] = values[digit]
        row = {key: row[key] for key in factors}
        if valid(row, platform):
            rows.append(row)
    return rows, {
        "space": size,
        "start": offset,
        "stop": stop,
        "next_offset": stop if stop < size else None,
        "excluded": stop - offset - len(rows),
    }


def inventory(source):
    """Fail when an advertised project switch lacks a declared test owner."""
    files = [source / "CMakeLists.txt", *sorted((source / "Cmake").rglob("*.cmake"))]
    text = "\n".join(path.read_text() for path in files)
    declared = set(re.findall(r"option\(\s*(LINALG_\w+)", text))
    declared.update(
        re.findall(r"set\(\s*(LINALG_\w+)\s+[^)]*?CACHE\s+(?:STRING|BOOL)", text)
    )
    covered = set(domains("linux-clang")) | set(FOCUSED)
    missing = declared - covered
    if missing:
        raise ValueError(f"CMake options missing CI coverage: {sorted(missing)}")
    return {
        key: FOCUSED.get(key, "pairwise/exhaustive build matrix")
        for key in sorted(declared)
    }


def generate(mode="pairwise", platform="all", offset=0, limit=256, batch_size=8):
    """Build the manifest and a bounded Actions job matrix."""
    source = Path(__file__).resolve().parents[2]
    manifest = {
        "mode": mode,
        "inventory": inventory(source),
        "platforms": {},
        "cases": {},
    }
    jobs = []
    selected = PLATFORMS if platform == "all" else [platform]
    for target in selected:
        factors = domains(target)
        if mode == "pairwise":
            rows, count = pairwise(factors, target)
            # Pairwise coverage alone does not guarantee the three switches
            # that activate cache + Icecream are enabled together.
            if not target.startswith("windows"):
                for backend in ("ccache", "sccache"):
                    row = complete(
                        factors,
                        {
                            "LINALG_ENABLE_ICECC": "ON",
                            "LINALG_ENABLE_CACHE": "ON",
                            "LINALG_CACHE_BACKEND": backend,
                            "LINALG_LTO_MODE": "off",
                            "BUILD_TESTING": "ON",
                            "LINALG_ENABLE_TESTING": "ON",
                            "LINALG_ENABLE_GTEST": "ON",
                        },
                        platform=target,
                    )
                    if row not in rows:
                        rows.append(row)
            info = {
                "feasible_pairs": count,
                "covered_pairs": len(set().union(*(pairs(row) for row in rows))),
                "cases": len(rows),
            }
            assert info["covered_pairs"] == count
        else:
            rows, info = exhaustive_window(factors, offset, limit, target)
            info["cases"] = len(rows)
        manifest["platforms"][target] = {**info, "domains": factors}
        ids = []
        for index, row in enumerate(rows):
            case_id = f"{target}-{index:04d}"
            manifest["cases"][case_id] = {"platform": target, "flags": row}
            ids.append(case_id)
        for batch in range(0, len(ids), batch_size):
            jobs.append(
                {
                    "os": PLATFORMS[target]["os"],
                    "platform": target,
                    "cc": PLATFORMS[target]["cc"],
                    "cxx": PLATFORMS[target]["cxx"],
                    "shard": f"{target}-{batch // batch_size:03d}",
                    "case_ids": ids[batch : batch + batch_size],
                }
            )
    if not jobs:
        raise ValueError(
            "The selected window contains no valid assignments; choose another offset"
        )
    if len(jobs) > 256:
        raise ValueError(
            "More than 256 jobs; reduce the exhaustive window or increase batch size"
        )
    manifest["matrix"] = {"include": jobs}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("pairwise", "exhaustive"), default="pairwise"
    )
    parser.add_argument("--platform", choices=("all", *PLATFORMS), default="all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.limit <= 2048 or args.batch_size < 1:
        parser.error(
            "offset must be nonnegative, limit 1..2048, and batch size positive"
        )
    manifest = generate(
        args.mode, args.platform, args.offset, args.limit, args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(
                "matrix=" + json.dumps(manifest["matrix"], separators=(",", ":")) + "\n"
            )
    print(
        json.dumps(
            {
                "jobs": len(manifest["matrix"]["include"]),
                "platforms": {
                    key: {k: v for k, v in value.items() if k != "domains"}
                    for key, value in manifest["platforms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
