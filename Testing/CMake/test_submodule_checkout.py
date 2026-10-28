"""Exercise CI submodule recovery and real Git checkouts without network access."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / ".github/actions/checkout-submodules/checkout.sh"

# Functions replace only the network commands and timer; run the real script
# with Bash's CI error handling, including failures inside the retry condition.
DRIVER = r'''
set -euo pipefail
sync_calls=0
update_calls=0
git() {
    if [[ "$*" == 'submodule sync --recursive' ]]; then
        sync_calls=$((sync_calls + 1))
        echo sync >> "$CHECKOUT_TRACE"
        ((sync_calls > FAIL_SYNCS))
    elif [[ "$*" == '-c protocol.version=2 submodule update --init --force --depth=1 --recursive' ]]; then
        update_calls=$((update_calls + 1))
        echo update >> "$CHECKOUT_TRACE"
        ((update_calls > FAIL_UPDATES))
    else
        echo "Unexpected git command: $*" >&2
        exit 99
    fi
}
sleep() {
    echo "sleep $1" >> "$CHECKOUT_TRACE"
}
source "$1"
'''


class SubmoduleCheckoutChecks(unittest.TestCase):
    def checkout(self, *, fail_updates=0, fail_syncs=0, script=SCRIPT):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace"
            result = subprocess.run(
                ["bash", "-c", DRIVER, "checkout-test", str(script)],
                env=dict(
                    os.environ,
                    CHECKOUT_TRACE=str(trace),
                    FAIL_UPDATES=str(fail_updates),
                    FAIL_SYNCS=str(fail_syncs),
                ),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, trace.read_text().splitlines()

    def test_success_does_not_retry_or_sleep(self):
        result, trace = self.checkout()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace, ["sync", "update"])

    def test_temporary_hosting_failure_recovers(self):
        result, trace = self.checkout(fail_updates=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            trace,
            ["sync", "update", "sleep 15", "sync", "update", "sleep 30", "sync", "update"],
        )

    def test_recovers_on_last_attempt(self):
        result, trace = self.checkout(fail_updates=4)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace.count("update"), 5)
        self.assertEqual(
            [call for call in trace if call.startswith("sleep")],
            ["sleep 15", "sleep 30", "sleep 60", "sleep 120"],
        )

    def test_persistent_failure_fails_the_job(self):
        result, trace = self.checkout(fail_updates=5)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(trace.count("update"), 5)
        self.assertEqual(trace[-1], "update")
        self.assertIn("::error::", result.stdout)

    def test_sync_failure_retries_before_downloading(self):
        result, trace = self.checkout(fail_syncs=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace, ["sync", "sleep 15", "sync", "update"])


def git(repository, *arguments):
    return subprocess.run(
        [
            "git", "-c", "protocol.file.allow=always",
            "-c", "user.name=Checkout test", "-c", "user.email=checkout@example.invalid",
            "-C", str(repository), *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def initialize(repository):
    repository.mkdir()
    git(repository, "init", "-q")


def commit(repository, version):
    (repository / "version").write_text(version)
    git(repository, "add", ".")
    git(repository, "commit", "-qm", version)
    return git(repository, "rev-parse", "HEAD")


class RealSubmoduleCheckoutChecks(unittest.TestCase):
    def test_windows_autocrlf_checkout_runs_the_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            initialize(repository)
            relative_script = SCRIPT.relative_to(SOURCE)
            script = repository / relative_script
            script.parent.mkdir(parents=True)
            script.write_bytes(SCRIPT.read_bytes())
            (repository / ".gitattributes").write_bytes(
                (SOURCE / ".gitattributes").read_bytes()
            )
            (repository / "line-ending-control.txt").write_bytes(b"first\nsecond\n")
            commit(repository, "helper")
            checkout = Path(directory) / "Windows checkout with spaces"
            git(repository, "-c", "core.autocrlf=true", "clone", str(repository), str(checkout))
            checked_out_script = checkout / relative_script
            self.assertIn(b"\r\n", (checkout / "line-ending-control.txt").read_bytes())
            self.assertNotIn(b"\r\n", checked_out_script.read_bytes())
            result, trace = SubmoduleCheckoutChecks().checkout(script=checked_out_script)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(trace, ["sync", "update"])

    def test_recursive_checkout_preserves_older_pinned_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            leaf, middle, repository = [base / name for name in ("leaf", "middle", "repository")]
            initialize(leaf)
            leaf_pin = commit(leaf, "pinned leaf")
            commit(leaf, "newer leaf")
            initialize(middle)
            commit(middle, "middle")
            git(middle, "submodule", "add", leaf.as_uri(), "nested")
            git(middle / "nested", "checkout", leaf_pin)
            middle_pin = commit(middle, "pinned middle")
            initialize(repository)
            commit(repository, "repository")
            git(repository, "submodule", "add", middle.as_uri(), "dependency")
            commit(repository, "pinned repository")
            commit(middle, "newer middle")
            checkout = base / "checkout with spaces"
            git(repository, "clone", str(repository), str(checkout))

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=checkout,
                env=dict(os.environ, GIT_ALLOW_PROTOCOL="file"),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git(checkout / "dependency", "rev-parse", "HEAD"), middle_pin)
            self.assertEqual(git(checkout / "dependency/nested", "rev-parse", "HEAD"), leaf_pin)
            self.assertEqual(git(checkout, "status", "--porcelain"), "")


if __name__ == "__main__":
    unittest.main()
