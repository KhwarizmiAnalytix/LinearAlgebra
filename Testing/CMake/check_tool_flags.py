"""Check every developer-tool switch combination using recording test doubles.

These are CMake integration checks, not claims that clang-tidy, IWYU, codespell,
Profiler, or GPU workloads were executed against the library.
"""

import itertools
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


class ToolFlagChecks(unittest.TestCase):
    """Keep option wiring independent of analyzer findings and optional accounts."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="linalg tool flags ")
        cls.work = Path(cls.temporary.name)
        recorder = cls.work / "recorder"
        recorder.mkdir()
        (recorder / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(RecordingTool LANGUAGES CXX)\n"
            f'add_executable(record-tool "{SOURCE.as_posix()}/Testing/CMake/record_tool.cpp")\n',
            encoding="utf-8",
        )
        build = recorder / "build"
        # A native executable preserves regex metacharacters in analyzer arguments.
        # A .cmd recorder sends them through cmd.exe and turns '|' into a pipe.
        for args in (
            ["cmake", "-S", str(recorder), "-B", str(build), "-G", "Ninja"],
            ["cmake", "--build", str(build)],
        ):
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        cls.tool = build / ("record-tool.exe" if os.name == "nt" else "record-tool")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, name, body, error=None):
        source = self.work / name
        source.mkdir(exist_ok=True)
        build = source / "build"
        text = (
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(ToolFlagFixture LANGUAGES CXX)\n"
            f'set(LINALG_SOURCE_DIR "{SOURCE.as_posix()}")\n'
            'set(LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/linalg")\n'
            'file(MAKE_DIRECTORY "${LINALG_BINARY_DIR}")\n' + body
        )
        (source / "CMakeLists.txt").write_text(text, encoding="utf-8")
        env = {**os.environ, "LINALG_CI_TOOL_LOG": str(source / "tool-log.jsonl")}
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if error:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(error, output)
        else:
            self.assertEqual(result.returncode, 0, output)
        return build, env

    def root(self, flags="", profiler=False):
        return (
            "add_library(Logging::Logging INTERFACE IMPORTED GLOBAL)\n"
            "set(BUILD_TESTING OFF)\nset(LINALG_ENABLE_TESTING OFF)\n"
            "set(CMAKE_DISABLE_FIND_PACKAGE_cpuinfo ON)\n"
            "set(CMAKE_DISABLE_FIND_PACKAGE_Profiler ON)\n"
            + (
                "add_library(Profiler::Profiler INTERFACE IMPORTED GLOBAL)\n"
                if profiler
                else ""
            )
            + flags
            + f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
        )

    def test_every_analyzer_switch_combination(self):
        keys = ("CLANGTIDY", "FIX", "IWYU", "SPELL")
        for values in itertools.product(("OFF", "ON"), repeat=len(keys)):
            with self.subTest(flags=dict(zip(keys, values))):
                flags = dict(zip(keys, values))
                body = "".join(
                    f"set(LINALG_ENABLE_{key} {value})\n"
                    for key, value in flags.items()
                )
                for name in (
                    "CLANG_TIDY_PATH",
                    "IWYU_EXECUTABLE",
                    "CODESPELL_EXECUTABLE",
                ):
                    body += f'set({name} "{self.tool.as_posix()}")\n'
                # Load data-bearing modules in the parent so probe targets inherit
                # the same arguments that the embedded LinearAlgebra target uses.
                for key, module in (("CLANGTIDY", "clang_tidy"), ("IWYU", "iwyu")):
                    if flags[key] == "ON":
                        body += (
                            f'include("${{LINALG_SOURCE_DIR}}/Cmake/{module}.cmake")\n'
                        )
                body = self.root(body)
                body += (
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int main() { return 0; }")\n'
                    'add_executable(tool_probe "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
                )
                if flags["CLANGTIDY"] == "ON":
                    body += (
                        "linalg_target_clang_tidy(tool_probe ${LINALG_ENABLE_FIX})\n"
                    )
                if flags["IWYU"] == "ON":
                    body += "linalg_apply_iwyu(tool_probe)\n"
                build, env = self.configure("tools-" + "-".join(values), body)
                targets = ["tool_probe"]
                if flags["SPELL"] == "ON":
                    targets.append("spell_check_" + SOURCE.name.lower())
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--target", *targets],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = build.parent / "tool-log.jsonl"
                calls = (
                    [json.loads(line) for line in log.read_text().splitlines()]
                    if log.exists()
                    else []
                )
                args = [arg for call in calls for arg in call]
                self.assertEqual(
                    "-warnings-as-errors=*" in args, flags["CLANGTIDY"] == "ON"
                )
                self.assertEqual(
                    "-fix" in args, flags["CLANGTIDY"] == flags["FIX"] == "ON"
                )
                self.assertEqual(
                    "--transitive_includes_only" in args, flags["IWYU"] == "ON"
                )
                self.assertEqual(
                    any(arg.startswith("--ignore-words=") for arg in args),
                    flags["SPELL"] == "ON",
                )

    def test_profiling_dependency_contract(self):
        for available in (False, True):
            with self.subTest(available=available):
                body = self.root(
                    "set(LINALG_ENABLE_PROFILING ON)\n", profiler=available
                )
                condition = "NOT " if available else ""
                body += f'if({condition}TARGET profile_linalg_matrix_operations)\nmessage(FATAL_ERROR "Profiler option ignored")\nendif()\n'
                self.configure(f"profiler-{available}", body)

    def test_examples_require_all_vendor_backends(self):
        self.configure(
            "invalid-examples",
            self.root("set(LINALG_BUILD_EXAMPLES ON)\n"),
            error="backend benchmark example requires BLAS",
        )

    @unittest.skipUnless(
        os.name == "nt" and os.environ.get("CXX", "").lower() in ("cl", "cl.exe"),
        "Native MSVC compiler contract",
    )
    def test_native_msvc_rejects_sanitizers(self):
        self.configure(
            "invalid-msvc-sanitizer",
            self.root("set(LINALG_ENABLE_SANITIZER ON)\n"),
            error="sanitizers require GCC or Clang",
        )

    def test_unsupported_lto_combinations(self):
        base = (
            'include("${LINALG_SOURCE_DIR}/Cmake/lto.cmake")\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int value;")\n'
            'add_library(probe STATIC "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            "set(LINALG_LTO_MODE auto)\n"
        )
        self.configure(
            "invalid-lto-cache",
            base
            + "set_property(TARGET probe PROPERTY LINALG_CACHE_BACKEND sccache)\n"
            + "linalg_target_apply_lto(probe LINALG)\n",
            error="LTO incompatible with sccache",
        )
        self.configure(
            "invalid-lto-cuda",
            base
            + "set(LINALG_ENABLE_CUBLAS ON)\n"
            + "linalg_target_apply_lto(probe LINALG)\n",
            error="LTO with CUDA is not supported",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
