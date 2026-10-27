"""Create CSV, Markdown, and charts from run_eigen_comparison.py artifacts."""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path


TITLES = {
    "multiplication_nn": "Multiplication NN",
    "multiplication_nt": "Multiplication NT",
    "multiplication_tn": "Multiplication TN",
    "multiplication_tt": "Multiplication TT",
    "lu": "LU factorization",
    "schur": "Complex Schur form and unitary vectors",
    "qr": "QR including explicit Q",
    "cholesky": "Cholesky lower factor",
    "svd": "Economy SVD",
    "symmetric_eigen": "Symmetric eigenpairs",
    "general_values": "General eigenvalues",
    "general_vectors": "General real eigenpairs",
    "transpose": "In-place transpose",
    "norm_frobenius": "Frobenius norm",
    "norm_one": "One norm",
    "norm_infinity": "Infinity norm",
    "linear_solve": "LU factorization and solve",
    "factorized_solve": "Solve with existing LU factors",
    "inversion": "LU factorization and inverse",
    "triangular_lower": "Lower triangular solve",
    "triangular_upper": "Upper triangular solve",
    "triangular_lower_transpose": "Transposed lower triangular solve",
    "triangular_upper_transpose": "Transposed upper triangular solve",
}


def identify(name):
    template, tail = name.split(">/", 1)
    function, arguments = template.split("<", 1)
    arguments = [part.strip() for part in arguments.split(",")]
    shape = tail.split("/min_time:", 1)[0]
    scalar = arguments[0]
    eigen = function.startswith("BM_Eigen")
    backend = "eigen" if eigen else arguments[1].rsplit("::", 1)[-1].lower()
    if "Multiplication" in function:
        flags = arguments[1:] if eigen else arguments[2:]
        flags = flags or ["false", "false"]
        mode = "".join("t" if flag == "true" else "n" for flag in flags)
        operation = "multiplication_" + mode
    elif function == "BM_EigenEigensolver":
        operation = (
            "symmetric_eigen"
            if arguments[1] == "false"
            else ("general_vectors" if arguments[2] == "true" else "general_values")
        )
    elif "SymmetricEigen" in function:
        operation = "symmetric_eigen"
    elif "GeneralEigen" in function:
        operation = "general_vectors" if arguments[-1] == "true" else "general_values"
    else:
        operation = next(
            key
            for token, key in [
                ("TriangularLowerTranspose", "triangular_lower_transpose"),
                ("TriangularUpperTranspose", "triangular_upper_transpose"),
                ("TriangularLower", "triangular_lower"),
                ("TriangularUpper", "triangular_upper"),
                ("FactorizedSolve", "factorized_solve"),
                ("LinearSolve", "linear_solve"),
                ("Inversion", "inversion"),
                ("Transpose", "transpose"),
                ("NormFrobenius", "norm_frobenius"),
                ("NormOne", "norm_one"),
                ("NormInfinity", "norm_infinity"),
                ("Cholesky", "cholesky"),
                ("Schur", "schur"),
                ("LU", "lu"),
                ("QR", "qr"),
                ("SVD", "svd"),
            ]
            if token in function
        )
    return operation, scalar, shape, backend


def summarize(directory, baseline=None, charts=True):
    """Write aligned results and optional native speedups against a prior run."""
    index = json.loads((directory / "runs.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = []
    failures = []
    for item in index:
        operation, scalar, shape, backend = identify(item["name"])
        if item["status"] != "ok":
            failures.append(
                {
                    "operation": operation,
                    "scalar": scalar,
                    "shape": shape,
                    "backend": backend,
                    "status": item["status"],
                    "log": item["log"],
                }
            )
            continue
        records = json.loads((directory / item["json"]).read_text())["benchmarks"]
        median = next(row for row in records if row.get("aggregate_name") == "median")
        cv = next(row for row in records if row.get("aggregate_name") == "cv")
        factor = {"ns": 0.001, "us": 1, "ms": 1000, "s": 1e6}[median["time_unit"]]
        rows.append(
            {
                "operation": operation,
                "scalar": scalar,
                "shape": shape,
                "backend": backend,
                "cpu_us": median["cpu_time"] * factor,
                "real_us": median["real_time"] * factor,
                "cpu_cv_percent": cv["cpu_time"] * 100,
                "repetitions": median["repetitions"],
                "raw_json": item["json"],
            }
        )
    (directory / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not rows:
        raise RuntimeError("No successful measurements; see failures.json")
    with (directory / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    backend_names = sorted({row["backend"] for row in rows} - {"native", "eigen"})
    backend_names = ["native", "eigen"] + backend_names
    for row in rows:
        groups.setdefault((row["operation"], row["scalar"], row["shape"]), {})[
            row["backend"]
        ] = row
    comparisons = []
    for (operation, scalar, shape), backends in groups.items():
        row = {"operation": operation, "scalar": scalar, "shape": shape}
        for backend in backend_names:
            row[backend + "_us"] = backends.get(backend, {}).get("cpu_us", "")
        eigen = backends.get("eigen", {}).get("cpu_us")
        for backend in backend_names:
            if backend == "eigen":
                continue
            timing = backends.get(backend, {}).get("cpu_us")
            row[backend + "_over_eigen"] = timing / eigen if timing and eigen else ""
        comparisons.append(row)
    comparisons.sort(
        key=lambda row: (
            list(TITLES).index(row["operation"]),
            row["scalar"],
            tuple(map(int, row["shape"].split("/"))),
        )
    )
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparisons[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)
    (directory / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    labels = {
        "native": "Native",
        "eigen": "Eigen",
        "blas_lapacke": "BLAS/LAPACKE",
        "mkl": "MKL",
    }
    header = " | ".join(labels.get(name, name) + " µs" for name in backend_names)

    def table_header(first):
        return [
            "| " + first + " | " + header + " | Native / Eigen |",
            "|" + "---|" * (len(first.split(" | ")) + len(backend_names) + 1),
        ]

    def table_row(row, title=False):
        prefix = (
            TITLES[row["operation"]] + " | " + row["shape"]
            if title
            else row["scalar"] + " / " + row["shape"]
        )
        values = [row[name + "_us"] for name in backend_names] + [
            row["native_over_eigen"]
        ]
        return (
            "| "
            + prefix
            + " | "
            + " | ".join(
                f"{value:.3f}" if isinstance(value, (float, int)) else "—"
                for value in values
            )
            + " |"
        )

    cache = metadata["cache"]
    test_count = "unknown"
    detail = directory / "tests-detail.log.gz"
    if detail.exists():
        counts = re.findall(
            r"\[  PASSED  \] (\d+) tests", gzip.decompress(detail.read_bytes()).decode()
        )
        if counts:
            test_count = counts[-1]
    lines = [
        "# Eigen comparison benchmark — " + metadata["started_utc"][:10],
        "",
        "; ".join(metadata["hardware"]) or metadata["os"],
        "",
        f"Release; CPU backend={cache.get('LINALG_CPU_BACKEND', 'unknown')}; "
        f"C++{cache.get('LINALG_CXX_STANDARD', 'unknown')}; LTO={cache.get('LINALG_LTO_MODE', 'unknown')}. "
        "All backends use one thread. Cases run sequentially with three repetitions, "
        "at least 0.1 seconds per repetition. Timings are median CPU microseconds.",
        "",
        f"**Tests:** {test_count} GoogleTest cases passed. See [test output](tests.log) "
        "and [detailed test output](tests-detail.log.gz).",
        "",
        f"**Benchmarks:** {len(rows)}/{len(index)} cases succeeded; {len(failures)} incomplete. "
        f"{len(comparisons)} operation/scalar/shape combinations.",
        "",
        "Native / Eigen is native time divided by Eigen time; values above 1 favor Eigen. "
        "A dash means no result was registered. BLAS/LAPACKE identifies the configured vendor library; "
        "norms use the shared CPU implementation for each CPU backend.",
        "",
        "## Largest registered cases",
        "",
    ]
    for scalar in ("float", "double"):
        lines += ["### " + scalar, ""] + table_header("Operation | Shape")
        for operation in TITLES:
            selected = [
                row
                for row in comparisons
                if row["operation"] == operation and row["scalar"] == scalar
            ]
            if selected:
                lines.append(
                    table_row(
                        max(
                            selected,
                            key=lambda row: tuple(map(int, row["shape"].split("/"))),
                        ),
                        True,
                    )
                )
        lines.append("")
    if baseline:
        previous = list(csv.DictReader((baseline / "timings.csv").open()))
        lookup = {
            (row["operation"], row["scalar"], row["shape"]): float(row["cpu_us"])
            for row in previous
            if row["backend"] == "native"
        }
        improvements = []
        for row in rows:
            key = (row["operation"], row["scalar"], row["shape"])
            if row["backend"] == "native" and key in lookup:
                improvements.append(
                    {
                        "operation": key[0],
                        "scalar": key[1],
                        "shape": key[2],
                        "before_us": lookup[key],
                        "after_us": row["cpu_us"],
                        "speedup": lookup[key] / row["cpu_us"],
                    }
                )
        if improvements:
            with (directory / "improvements.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(improvements[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(improvements)
            lines += [
                "## Native changes versus baseline",
                "",
                f"Baseline: `{baseline.name}`. "
                "Speedup is baseline time divided by current time. Check both runs' metadata "
                "for configuration and source differences. Full results: [improvements.csv](improvements.csv).",
                "",
                "| Operation | Scalar | Shape | Before µs | After µs | Speedup |",
                "|---|---|---|---:|---:|---:|",
            ]
            for operation in TITLES:
                for scalar in ("float", "double"):
                    selected = [
                        row
                        for row in improvements
                        if row["operation"] == operation and row["scalar"] == scalar
                    ]
                    if selected:
                        row = max(
                            selected,
                            key=lambda row: tuple(map(int, row["shape"].split("/"))),
                        )
                        lines.append(
                            f"| {TITLES[operation]} | {scalar} | {row['shape']} | "
                            f"{row['before_us']:.3f} | {row['after_us']:.3f} | {row['speedup']:.2f}× |"
                        )
            lines.append("")
    if charts:
        make_charts(directory, comparisons, backend_names, labels)
    if all(
        (directory / f"scaling-{scalar}.png").exists() for scalar in ("float", "double")
    ):
        lines += [
            "## Scaling",
            "",
            "Square inputs; triangular solves fix the number of right-hand sides at 16.",
            "",
            "![Float scaling](scaling-float.png)",
            "",
            "![Double scaling](scaling-double.png)",
            "",
        ]
    lines += [
        "## Measurement details",
        "",
        "- Compiler: `" + metadata["compiler"].splitlines()[0] + "`.",
        "- Configuration, source hashes, timestamps, and thread settings: [metadata.json](metadata.json).",
        "- All timings and variation: [timings.csv](timings.csv); aligned results: [comparison.csv](comparison.csv).",
        "- Commands and outcomes: [runs.json](runs.json); original JSON and logs: [raw/](raw/).",
        "- Measurement phases and source snapshots are recorded in metadata. When cases are repeated after a final change, their earlier measurements remain in `first-pass/`.",
        f"- Native LU pivoting: {cache.get('LINALG_LU_PIVOTING', 'unknown')}. Eigen LU uses partial pivoting. "
        "The common diagonally dominant input avoids swaps.",
        "- QR includes explicit Q formation. SVD includes thin U and row-major VT; algorithms differ by backend.",
        "- Input refresh is untimed except for SVD. Outputs and workspaces are preallocated; internal backend allocations and copies remain timed.",
        "- Factorized solves exclude LU factorization for every backend. Linear solve and inversion include factorization.",
        "- Triangular solves include copying the right-hand side and use a non-unit diagonal. Transpose is in place for every backend.",
        "- General eigenpairs use a real-spectrum input distinct from eigenvalues-only cases.",
        "- Schur uses dense nonsymmetric real inputs, full complex triangular S and unitary Q, "
        "and split real/imaginary output planes. Eigen input promotion and output splitting are timed. "
        "Reconstruction, unitarity, and triangular form are checked before timing. "
        "Native and BLAS/MKL Schur currently share the portable algorithm.",
        "- Native SVD registrations stop at 128. Results describe these inputs on this host; scheduling and thermal state can affect them.",
        "",
        "## All measurements",
        "",
    ]
    for operation, title in TITLES.items():
        selected = [row for row in comparisons if row["operation"] == operation]
        if selected:
            lines += ["### " + title, ""] + table_header("Scalar / shape")
            lines.extend(table_row(row) for row in selected)
            lines.append("")
    if failures:
        lines += [
            "## Incomplete cases",
            "",
            "| Operation | Scalar | Shape | Backend | Outcome |",
            "|---|---|---|---|---|",
        ]
        for row in failures:
            lines.append(
                f"| {row['operation']} | {row['scalar']} | {row['shape']} | {row['backend']} | [{row['status']}]({row['log']}) |"
            )
    (directory / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n")
    print(
        f"Report: {directory / 'REPORT.md'} ({len(rows)} completed, {len(failures)} incomplete)"
    )


def make_charts(directory, rows, backends, labels):
    """Export scaling charts for each measured operation and scalar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    operations = [
        operation
        for operation in TITLES
        if any(row["operation"] == operation for row in rows)
    ]
    height = (len(operations) + 1) // 2
    for scalar in ("float", "double"):
        fig, axes = plt.subplots(
            height, 2, figsize=(12, 3 * height), layout="constrained", squeeze=False
        )
        for ax, operation in zip(axes.flat, operations):
            for backend in backends:
                selected = [
                    row
                    for row in rows
                    if row["operation"] == operation
                    and row["scalar"] == scalar
                    and row[backend + "_us"] != ""
                    and (
                        len(set(row["shape"].split("/"))) == 1
                        or (
                            operation.startswith("triangular_")
                            and row["shape"].endswith("/16")
                        )
                    )
                ]
                selected.sort(key=lambda row: int(row["shape"].split("/")[0]))
                if selected:
                    ax.plot(
                        [int(row["shape"].split("/")[0]) for row in selected],
                        [row[backend + "_us"] for row in selected],
                        "o-",
                        label=labels.get(backend, backend),
                    )
            ax.set(
                xscale="log",
                yscale="log",
                title=TITLES[operation],
                xlabel="Matrix order n",
                ylabel="Median CPU time (µs)",
            )
            ax.grid(True, which="both", alpha=0.18)
            if ax.lines:
                ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(operations) :]:
            ax.set_visible(False)
        fig.suptitle(f"Native and Eigen comparison — {scalar} · Release · one thread")
        fig.savefig(directory / f"scaling-{scalar}.png", dpi=130)
        fig.savefig(directory / f"scaling-{scalar}.svg")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--baseline", type=Path, help="Prior result directory containing timings.csv"
    )
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    summarize(args.directory, args.baseline, not args.no_charts)
