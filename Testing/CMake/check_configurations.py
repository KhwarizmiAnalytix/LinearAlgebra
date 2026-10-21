"""Regression checks for CMake configuration contracts; no network downloads."""

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


class ConfigurationChecks(unittest.TestCase):
    """Exercise embedding, failure recovery, tooling, and dependency boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="linalg-cmake-checks-")
        cls.work = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, name, body, options=(), error=None):
        """Configure a small host and check its result and expected diagnostic."""
        source = self.work / name
        source.mkdir(exist_ok=True)
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(ConfigurationProbe LANGUAGES CXX)\n"
            f'set(LINALG_SOURCE_DIR "{SOURCE.as_posix()}")\n'
            'set(LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")\n' + body,
            encoding="utf-8",
        )
        build = source / "build"
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_DISABLE_FIND_PACKAGE_cpuinfo=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Profiler=ON",
            *options,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        output = result.stdout + result.stderr
        if error:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(error, output)
        else:
            self.assertEqual(result.returncode, 0, output)
        return build

    def host(self, before="", after=""):
        """Supply an imported Logging target to exercise target reuse."""
        return (
            "enable_testing()\n"
            "add_library(Logging::Logging INTERFACE IMPORTED GLOBAL)\n"
            + before
            + f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            + after
        )

    def test_embedding_and_test_switches(self):
        for name, options, has_gtest, count in [
            ("host-default", [], False, 0),
            ("host-testing-off", ["-DBUILD_TESTING=OFF"], False, 0),
            (
                "host-gtest-off",
                ["-DBUILD_TESTING=ON", "-DLINALG_ENABLE_GTEST=OFF"],
                False,
                1,
            ),
            ("host-gtest-on", ["-DBUILD_TESTING=ON"], True, 2),
        ]:
            with self.subTest(name=name):
                assertion = "if(NOT TARGET gtest)" if has_gtest else "if(TARGET gtest)"
                build = self.configure(
                    name,
                    self.host(
                        after=assertion
                        + '\nmessage(FATAL_ERROR "GTest switch ignored")\nendif()\n'
                    ),
                    options,
                )
                result = subprocess.run(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                tests = json.loads(result.stdout)["tests"]
                self.assertEqual(len(tests), count)
                if not has_gtest and count:
                    subprocess.run(
                        ["ctest", "--test-dir", str(build), "--output-on-failure"],
                        check=True,
                    )

    def test_bundled_logging_preserves_parent_cache(self):
        build = self.configure(
            "bundled-logging",
            'set(BUILD_TESTING ON CACHE BOOL "Host testing")\n'
            'set(LOGGING_ENABLE_TESTING ON CACHE BOOL "Host logging choice")\n'
            "set(LINALG_ENABLE_TESTING OFF)\n"
            f'add_subdirectory("{SOURCE.as_posix()}" linalg)\n'
            "if(NOT BUILD_TESTING OR NOT LOGGING_ENABLE_TESTING OR TARGET LoggingCxxTests)\n"
            'message(FATAL_ERROR "Dependency configuration changed host choices")\nendif()\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp" '
            '"#include \\"include/matrix_operation/trace.h\\"\\n'
            "int main() { double a[] = {1, 2, 3, 4}; "
            'return linalg::trace<double, linalg::linalg_backend::NATIVE>::apply(a, 2, 2) == 5 ? 0 : 1; }")\n'
            'add_executable(consumer "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp")\n'
            "target_link_libraries(consumer PRIVATE LinearAlgebra::LinearAlgebra)\n"
            'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-path.txt" '
            'CONTENT "$<TARGET_FILE:consumer>")\n',
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "consumer", "--parallel", "4"],
            check=True,
        )
        subprocess.run([(build / "consumer-path.txt").read_text()], check=True)

    def test_owned_modules_cannot_be_shadowed(self):
        shadow = self.work / "shadow"
        shadow.mkdir()
        for module in (SOURCE / "Cmake").glob("*.cmake"):
            (shadow / module.name).write_text(
                'message(FATAL_ERROR "Loaded host module")\n'
            )
        self.configure(
            "module-shadow",
            self.host(
                before=f'list(PREPEND CMAKE_MODULE_PATH "{shadow.as_posix()}")\n'
            ),
        )

    def test_required_validation_repeats_after_failure(self):
        body = 'include("${LINALG_SOURCE_DIR}/Cmake/checks.cmake")\n'
        for _ in range(2):
            self.configure(
                "failed-validation",
                body,
                ["-DCMAKE_DISABLE_FIND_PACKAGE_Threads=ON"],
                error="REQUIRED",
            )

    def test_invalid_isa_and_lto_fail_at_configure(self):
        for option, error in [
            ("-DLINALG_VECTORIZATION_ISA=typo", "Invalid LINALG_VECTORIZATION_ISA"),
            ("-DLINALG_LTO_MODE=typo", "Invalid LINALG_LTO_MODE"),
            ("-DLINALG_LINKER_CHOICE=typo", "Invalid linker choice"),
        ]:
            with self.subTest(option=option):
                self.configure(
                    error.replace(" ", "-"), self.host(), [option], error=error
                )

    def test_missing_cache_is_rejected(self):
        body = (
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int x;")\n'
            'add_library(probe STATIC "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/cache.cmake")\n'
            "set(ENABLE_CACHE ON)\nset(CACHE_BACKEND ccache)\n"
            'set(LINALG_CCACHE_PROGRAM "${CMAKE_CURRENT_BINARY_DIR}/missing-ccache")\n'
            "linalg_target_apply_cache(probe ENABLE_CACHE CACHE_BACKEND)\n"
        )
        self.configure("missing-cache", body, error="requested cache ccache not found")
        unavailable = body.replace(
            'set(LINALG_CCACHE_PROGRAM "${CMAKE_CURRENT_BINARY_DIR}/missing-ccache")',
            "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH OFF)\n"
            "set(CMAKE_FIND_USE_CMAKE_PATH OFF)\n"
            "set(CMAKE_FIND_USE_INSTALL_PREFIX OFF)\n"
            "set(LINALG_CCACHE_PROGRAM LINALG_CCACHE_PROGRAM-NOTFOUND)",
        )
        self.configure(
            "notfound-cache", unavailable, error="requested cache ccache not found"
        )

    @unittest.skipUnless(os.name == "posix", "Icecream uses POSIX compiler wrappers")
    def test_icecream_and_compiler_caches_compose(self):
        for backend in ("sccache", "ccache"):
            with self.subTest(backend=backend):
                tools = self.work / f"{backend} tools' directory"
                tools.mkdir()
                log = tools / "commands.jsonl"
                cache = tools / backend
                icecc = tools / "icecc"
                script = (
                    "#!/usr/bin/env python3\n"
                    "import json, os, subprocess, sys\n"
                    f"with open({str(log)!r}, 'a') as log:\n"
                    "    log.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + '\\n')\n"
                )
                for executable in (cache, icecc):
                    executable.write_text(
                        script
                        + "sys.exit(subprocess.run(sys.argv[1:], check=False).returncode)\n"
                    )
                cache.chmod(0o755)
                icecc.chmod(0o755)
                body = (
                    "enable_language(C)\n"
                    'include("${LINALG_SOURCE_DIR}/Cmake/cache.cmake")\n'
                    'include("${LINALG_SOURCE_DIR}/Cmake/helper_macros.cmake")\n'
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/answer.c" "int answer(void) { return 42; }")\n'
                    'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" '
                    '"extern \\"C\\" int answer(void); int main() { return answer() == 42 ? 0 : 1; }")\n'
                    'add_executable(probe "${CMAKE_CURRENT_BINARY_DIR}/answer.c" "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")\n'
                    "set(ENABLE_CACHE ON)\nset(ENABLE_ICECC ON)\n"
                    f"set(CACHE_BACKEND {backend})\n"
                    f'set(LINALG_{backend.upper()}_PROGRAM "{cache.as_posix()}")\n'
                    f'set(LINALG_ICECC_EXECUTABLE "{icecc.as_posix()}")\n'
                    "linalg_target_apply_cache(probe ENABLE_CACHE CACHE_BACKEND)\n"
                    "linalg_target_optional_icecc(probe ENABLE_ICECC)\n"
                    'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-path.txt" CONTENT "$<TARGET_FILE:probe>")\n'
                )
                build = self.configure(f"icecc-{backend}", body)
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--parallel", "2"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                subprocess.run([(build / "probe-path.txt").read_text()], check=True)
                commands = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(sum(command[0] == backend for command in commands), 2)
                self.assertEqual(sum(command[0] == "icecc" for command in commands), 2)
                wrapper = build / "CMakeFiles/probe-launchers/icecc-CXX-compiler"
                previous_wrapper = wrapper.read_text()
                icecc.write_text(
                    icecc.read_text() + "\n# Updated Icecream executable\n"
                )
                regenerated = subprocess.run(
                    ["cmake", "-S", str(build.parent), "-B", str(build)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    regenerated.returncode, 0, regenerated.stdout + regenerated.stderr
                )
                self.assertNotEqual(previous_wrapper, wrapper.read_text())
                (build / "main.cpp").write_text(
                    "#error compiler failure must propagate\n"
                )
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--parallel", "2"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "compiler failure must propagate", result.stdout + result.stderr
                )
                if backend == "sccache":
                    self.configure(
                        "icecc-sccache-lto",
                        body
                        + 'include("${LINALG_SOURCE_DIR}/Cmake/lto.cmake")\n'
                        + "set(LINALG_LTO_MODE thin)\n"
                        + "linalg_target_apply_lto(probe LINALG)\n",
                        error="LTO incompatible with sccache",
                    )

    @unittest.skipIf(
        os.name == "nt" and os.environ.get("CXX", "").lower() in ("cl", "cl.exe"),
        "Native MSVC does not support the GCC/Clang sanitizer helper",
    )
    def test_sanitizer_regenerates_assets_and_exports_runtime(self):
        body = (
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int x;")\n'
            'add_library(probe STATIC "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/sanitize.cmake")\n'
            "set(LINALG_SANITIZER_TYPE address)\n"
            "linalg_configure_sanitizer(probe LINALG)\n"
            "get_target_property(runtime probe INTERFACE_LINK_OPTIONS)\n"
            'if(NOT runtime MATCHES "fsanitize=address")\n'
            'message(FATAL_ERROR "Missing sanitizer runtime requirement")\nendif()\n'
            'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compiler.txt" "${CMAKE_CXX_COMPILER_ID}")\n'
        )
        build = self.configure("sanitizer-regeneration", body)
        if "Clang" in (build / "compiler.txt").read_text():
            ignore = build / "sanitizer_ignore.txt"
            self.assertTrue(ignore.is_file())
            ignore.unlink()
            self.configure("sanitizer-regeneration", body)
            self.assertTrue(ignore.is_file())

    def test_benchmark_fetchcontent_and_vendored_paths(self):
        body = (
            'include("${LINALG_SOURCE_DIR}/Cmake/LinearAlgebraDependencies.cmake")\n'
            "linalg_setup_benchmark()\n"
        )
        self.configure("benchmark-vendored", body)
        self.configure(
            "benchmark-fetchcontent",
            body,
            [f"-DFETCHCONTENT_SOURCE_DIR_BENCHMARK={SOURCE / 'ThirdParty/benchmark'}"],
        )

    def test_dependency_helper_guards_and_flag_filter(self):
        self.configure(
            "dependency-helper",
            'include("${LINALG_SOURCE_DIR}/Cmake/third_party_helpers.cmake")\n'
            "linalg_add_third_party_library(skipped CONDITION OFF LIBRARY_TYPE HEADER_ONLY)\n"
            'if(TARGET skipped)\nmessage(FATAL_ERROR "Condition ignored")\nendif()\n'
            "linalg_add_third_party_library(headers LIBRARY_TYPE HEADER_ONLY "
            'INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" AUTO_ADD_TO_DEPENDENCIES OFF)\n'
            "get_property(deps GLOBAL PROPERTY _LINALG_DEPENDENCY_LIBS)\n"
            'if(deps)\nmessage(FATAL_ERROR "Auto dependency switch ignored")\nendif()\n'
            'linalg_filter_flags_internal("-Wall -O2 -g" "-W.*" filtered)\n'
            'if(NOT filtered STREQUAL "-O2 -g")\n'
            'message(FATAL_ERROR "Flag filter removed non-warning flags")\nendif()\n',
        )

    def test_valgrind_dashboard_and_recursive_timeouts(self):
        child = self.work / "valgrind-child"
        child.mkdir()
        (child / "CMakeLists.txt").write_text(
            'add_test(NAME probe COMMAND "${CMAKE_COMMAND}" -E true)\n'
            "set_tests_properties(probe PROPERTIES TIMEOUT 2)\n"
        )
        build = self.configure(
            "valgrind-dashboard",
            "set(BUILD_TESTING ON)\n"
            'set(MEMORYCHECK_COMMAND "${CMAKE_COMMAND}")\n'
            'include("${LINALG_SOURCE_DIR}/Cmake/valgrind.cmake")\n'
            "include(CTest)\n"
            f'add_subdirectory("{child.as_posix()}" child)\n'
            'linalg_apply_valgrind_timeouts("${CMAKE_CURRENT_SOURCE_DIR}")\n',
        )
        dashboard = (build / "DartConfiguration.tcl").read_text()
        self.assertIn("--error-exitcode=1", dashboard)
        self.assertIn("valgrind_suppression.txt", dashboard)
        result = subprocess.run(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
            capture_output=True,
            text=True,
            check=True,
        )
        properties = json.loads(result.stdout)["tests"][0]["properties"]
        self.assertIn({"name": "TIMEOUT", "value": 40.0}, properties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    unittest.main(argv=[__file__], verbosity=2)
