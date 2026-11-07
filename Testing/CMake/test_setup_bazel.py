"""Launcher composition for Scripts/setup_bazel.py cache and Icecream chains."""

import contextlib
import importlib.util
import io
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "Scripts"
SPEC = importlib.util.spec_from_file_location(
    "setup_bazel", SCRIPTS / "setup_bazel.py"
)
setup_bazel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_bazel)


def _stub(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class BazelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.records = self.root / "records"
        self.records.mkdir()
        repo = patch.object(setup_bazel, "_repo_root", return_value=str(self.root))
        repo.start()
        self.addCleanup(repo.stop)

    def configuration(self):
        return setup_bazel.LinearAlgebraBazelConfiguration([])

    def which(self, programs):
        def resolve(name):
            return programs.get(name)

        return patch.object(setup_bazel.shutil, "which", side_effect=resolve)

    def test_cache_and_icecc_use_one_compiler_wrapper(self):
        compiler_dir = self.root / "o'brien tools"
        compiler_dir.mkdir()
        compiler = compiler_dir / "clang"
        cache = self.bin / "ccache"
        icecc = self.bin / "icecc"
        _stub(
            cache,
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > '{self.records / 'cache'}'\n"
            'tool="$1"\n'
            "shift\n"
            'exec "$tool" "$@"\n',
        )
        _stub(
            icecc,
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$0\" \"$@\" > '{self.records / 'icecc'}'\n"
            'tool="$1"\n'
            "shift\n"
            'exec "$tool" "$@"\n',
        )
        _stub(
            compiler,
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$0\" \"$@\" > '{self.records / 'compiler'}'\n",
        )
        config = self.configuration()
        config.cache_backend = "ccache"
        config.icecc = True
        programs = {
            "ccache": str(cache),
            "icecc": str(icecc),
            str(compiler): str(compiler),
        }
        with self.which(programs):
            launcher = config._write_compiler_launcher(str(compiler), "cxx")

        self.assertIsNotNone(launcher)
        text = Path(launcher).read_text(encoding="utf-8")
        wrapper = self.root / ".bazel_launchers" / "icecc_cxx_compiler.sh"
        self.assertIn(setup_bazel._shell_single_quote(str(cache)), text)
        self.assertIn(setup_bazel._shell_single_quote(str(wrapper)), text)
        self.assertNotIn(str(icecc), text)
        wrapper_text = wrapper.read_text(encoding="utf-8")
        self.assertIn(setup_bazel._shell_single_quote(str(icecc)), wrapper_text)
        self.assertIn(setup_bazel._shell_single_quote(str(compiler)), wrapper_text)

        if os.name == "nt":
            return
        result = subprocess.run(
            [launcher, "-c", "hello.cpp"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.records / "cache").read_text(encoding="utf-8").splitlines(),
            [str(wrapper), "-c", "hello.cpp"],
        )
        self.assertEqual(
            (self.records / "icecc").read_text(encoding="utf-8").splitlines(),
            [str(icecc), str(compiler), "-c", "hello.cpp"],
        )
        self.assertEqual(
            (self.records / "compiler").read_text(encoding="utf-8").splitlines(),
            [str(compiler), "-c", "hello.cpp"],
        )

    def test_single_tool_invokes_the_real_compiler_directly(self):
        compiler = self.bin / "clang"
        cache = self.bin / "ccache"
        _stub(compiler, "#!/bin/sh\nexit 0\n")
        _stub(cache, "#!/bin/sh\nexit 0\n")
        config = self.configuration()
        config.cache_backend = "ccache"
        config.icecc = False
        with self.which({"ccache": str(cache), str(compiler): str(compiler)}):
            launcher = config._write_compiler_launcher(str(compiler), "cc")
        text = Path(launcher).read_text(encoding="utf-8")
        self.assertIn(setup_bazel._shell_single_quote(str(cache)), text)
        self.assertIn(setup_bazel._shell_single_quote(str(compiler)), text)
        wrapper = self.root / ".bazel_launchers" / "icecc_cc_compiler.sh"
        self.assertFalse(wrapper.exists())


if __name__ == "__main__":
    unittest.main()
