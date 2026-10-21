"""Verify coverage guarantees and exhaustive-window bookkeeping independently."""

import importlib.util
import itertools
import math
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flag_matrix", SOURCE / ".github/scripts/flag_matrix.py"
)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class FlagMatrixChecks(unittest.TestCase):
    """Exercise constrained coverage, reproducibility, windows, and inventory."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = matrix.generate()

    def test_all_platforms_cover_every_feasible_pair(self):
        self.assertEqual(set(self.manifest["platforms"]), set(matrix.PLATFORMS))
        for platform, info in self.manifest["platforms"].items():
            with self.subTest(platform=platform):
                factors = matrix.domains(platform)
                rows = [
                    case["flags"]
                    for case in self.manifest["cases"].values()
                    if case["platform"] == platform
                ]
                covered = set().union(*(matrix.pairs(row) for row in rows))
                for row in rows:
                    self.assertTrue(matrix.valid(row, platform))
                    self.assertEqual(set(row), set(factors))
                    for key, value in row.items():
                        self.assertIn(value, factors[key])
                expected = set()
                for a, b in itertools.combinations(sorted(factors), 2):
                    for av, bv in itertools.product(factors[a], factors[b]):
                        # Enumerate just the constraint-bearing dimensions. Other
                        # dimensions are independent, so no full-space walk is needed.
                        partial = {a: av, b: bv}
                        constrained = [
                            key
                            for key in (
                                "LINALG_ENABLE_CACHE",
                                "LINALG_CACHE_BACKEND",
                                "LINALG_ENABLE_ICECC",
                                "LINALG_LTO_MODE",
                                "LINALG_ENABLE_COVERAGE",
                                "LINALG_ENABLE_SANITIZER",
                                "LINALG_LINKER_CHOICE",
                            )
                            if key not in partial
                        ]
                        if any(
                            matrix.valid(
                                dict(partial, **dict(zip(constrained, values))),
                                platform,
                            )
                            for values in itertools.product(
                                *(factors[key] for key in constrained)
                            )
                        ):
                            expected.add(((a, av), (b, bv)))
                self.assertEqual(covered, expected)
                self.assertEqual(len(expected), info["feasible_pairs"])

    def test_generation_is_deterministic_and_shards_partition_cases(self):
        self.assertEqual(self.manifest, matrix.generate())
        jobs = self.manifest["matrix"]["include"]
        self.assertLessEqual(len(jobs), 256)
        case_ids = [case for job in jobs for case in job["case_ids"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertEqual(set(case_ids), set(self.manifest["cases"]))
        self.assertTrue(all(1 <= len(job["case_ids"]) <= 8 for job in jobs))

    def test_icecream_cache_regressions_are_active_together(self):
        for platform in ("linux-gcc", "linux-clang", "macos-clang"):
            for backend in ("ccache", "sccache"):
                self.assertTrue(
                    any(
                        case["platform"] == platform
                        and case["flags"]["LINALG_ENABLE_ICECC"] == "ON"
                        and case["flags"]["LINALG_ENABLE_CACHE"] == "ON"
                        and case["flags"]["LINALG_CACHE_BACKEND"] == backend
                        and case["flags"]["LINALG_LTO_MODE"] == "off"
                        and case["flags"]["BUILD_TESTING"] == "ON"
                        and case["flags"]["LINALG_ENABLE_TESTING"] == "ON"
                        for case in self.manifest["cases"].values()
                    )
                )

    def test_exhaustive_windows_match_cartesian_enumeration(self):
        factors = {
            "LINALG_ENABLE_CACHE": matrix.BOOL,
            "LINALG_CACHE_BACKEND": ("ccache", "sccache", "buildcache"),
            "LINALG_ENABLE_ICECC": matrix.BOOL,
            "LINALG_LTO_MODE": ("off", "ipo"),
            "LINALG_ENABLE_COVERAGE": matrix.BOOL,
            "LINALG_ENABLE_SANITIZER": matrix.BOOL,
        }
        all_rows = [
            dict(zip(factors, values))
            for values in itertools.product(*factors.values())
        ]
        collected = []
        size = math.prod(map(len, factors.values()))
        for offset in range(0, size, 7):
            rows, ledger = matrix.exhaustive_window(factors, offset, 7)
            expected = [
                row for row in all_rows[offset : offset + 7] if matrix.valid(row)
            ]
            self.assertEqual(rows, expected)
            self.assertEqual(ledger["excluded"], min(7, size - offset) - len(rows))
            self.assertEqual(
                ledger["next_offset"], offset + 7 if offset + 7 < size else None
            )
            collected.extend(rows)
        self.assertEqual(collected, [row for row in all_rows if matrix.valid(row)])
        with self.assertRaises(ValueError):
            matrix.exhaustive_window(factors, size, 7)

    def test_constraints_preserve_normalized_and_instrumented_cases(self):
        base = {
            "LINALG_ENABLE_CACHE": "ON",
            "LINALG_CACHE_BACKEND": "sccache",
            "LINALG_LTO_MODE": "full",
            "LINALG_ENABLE_COVERAGE": "OFF",
            "LINALG_ENABLE_SANITIZER": "OFF",
        }
        self.assertFalse(matrix.valid(base))
        for override in (
            {"LINALG_ENABLE_CACHE": "OFF"},
            {"LINALG_CACHE_BACKEND": "none"},
            {"LINALG_LTO_MODE": "off"},
            {"LINALG_ENABLE_COVERAGE": "ON"},
            {"LINALG_ENABLE_SANITIZER": "ON"},
        ):
            self.assertTrue(matrix.valid({**base, **override}))
        self.assertFalse(
            matrix.valid(
                {
                    "LINALG_ENABLE_CACHE": "ON",
                    "LINALG_CACHE_BACKEND": "buildcache",
                    "LINALG_ENABLE_ICECC": "ON",
                }
            )
        )
        gcc_lto = {
            **base,
            "LINALG_CACHE_BACKEND": "none",
            "LINALG_LINKER_CHOICE": "lld",
        }
        self.assertFalse(matrix.valid(gcc_lto, "linux-gcc"))
        self.assertTrue(matrix.valid(gcc_lto, "linux-clang"))
        self.assertTrue(
            matrix.valid({"LINALG_ENABLE_BLIS": "ON", "LINALG_ENABLE_BLAS": "OFF"})
        )

    def test_inventory_rejects_uncovered_boolean_and_enum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for declaration in (
                'option(LINALG_NEW_FEATURE "feature" OFF)',
                'set(LINALG_NEW_MODE "off" CACHE STRING "mode")',
            ):
                (root / "CMakeLists.txt").write_text(declaration)
                with self.assertRaisesRegex(ValueError, "missing CI coverage"):
                    matrix.inventory(root)
            (root / "CMakeLists.txt").write_text("")
            (root / "Cmake").mkdir()
            (root / "Cmake/feature.cmake").write_text(
                'option(LINALG_NEW_MODULE_FEATURE "feature" OFF)'
            )
            with self.assertRaisesRegex(ValueError, "missing CI coverage"):
                matrix.inventory(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
