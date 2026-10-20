"""Check provider-aware reports using recorded data as synthetic fixtures."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
RECORDED = ROOT / "results/2026-10-19"


class ReportTest(unittest.TestCase):
    def prepare(self, directory, provider="openblas", include_openblas=False):
        data = directory / "data"
        data.mkdir()
        for name in (
            "results.csv",
            "rerun_results.csv",
            "gpu_steady_results.csv",
            "transfers.csv",
        ):
            with (RECORDED / name).open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            if name == "results.csv":
                converted = []
                for row in rows:
                    if row["backend"] == "openblas":
                        if include_openblas:
                            converted.append(dict(row))
                        row["backend"] = provider
                    converted.append(row)
                rows = converted
            elif name != "transfers.csv":
                # Cohort sizes vary between real measurement sets.
                rows = []
            with (data / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        metadata = {
            "date": "synthetic-provider-test",
            "library_commit": "test-fixture",
            "cpu_environment": {"BLIS_NUM_THREADS": "1"},
        }
        (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return data

    def generate(self, data, output):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "report.py"),
                "--data-dir",
                str(data),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def test_provider_columns_and_charts(self):
        for provider, include_openblas in (
            ("openblas", False),
            ("blis", False),
            ("blis", True),
        ):
            with self.subTest(provider=provider, mixed=include_openblas):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    data = self.prepare(directory, provider, include_openblas)
                    output = directory / "report"
                    result = self.generate(data, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = (output / "analysis.md").read_text(encoding="utf-8")
                    labels = ["Native"]
                    if provider == "openblas" or include_openblas:
                        labels.append("OpenBLAS")
                    if provider == "blis":
                        labels.append("BLIS")
                    labels.extend(["MKL", "GPU"])
                    self.assertIn("| Operation | " + " | ".join(labels) + " |", report)
                    self.assertIn("synthetic-provider-test", report)
                    self.assertIn("0 cases have longer-sample measurements", report)
                    self.assertNotIn("At library commit 15534ab", report)
                    for chart in ("scaling.png", "gemm_transfers.png"):
                        self.assertGreater(
                            (output / "assets" / chart).stat().st_size, 0
                        )
                    with (data / "final_results.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertIn(provider, {row["backend"] for row in rows})
                    self.assertTrue(
                        all(row["measurement_source"] == "initial" for row in rows)
                    )

    def test_unknown_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = self.prepare(directory, "unknown_provider")
            result = self.generate(data, directory / "report")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown backend", result.stderr)


if __name__ == "__main__":
    unittest.main()
