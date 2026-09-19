---
type: "always_apply"
---

# Build and verification

Read `CLAUDE.md` and the `project-build` skill for this repository's actual
build commands. Use its existing setup helpers where provided. Python
packages use their packaging and pytest workflow instead of a C++ build.

Configure before the first build and after changing build options or source
registration. Run tests for changed behavior. Where both CMake and Bazel
exist, keep their affected source lists, tests, and options consistent and
validate both for non-trivial source or build changes.

Use the `session-checklist` skill before handing off non-trivial work or
creating a commit. For guidance-only edits, check links, syntax, and the
diff; compilation is not required. Report what actually ran and any blockers.
