---
name: session-checklist
description: Verify non-trivial changes in Logging before handoff or commit, including supported build systems, tests, configured lint, and diff review.
---

# session-checklist

Read [CLAUDE.md](../../../CLAUDE.md) and review the working diff without
discarding unrelated user changes. Scale the checks to what changed.

1. **Build definitions.** For source or build changes, compare the affected
   CMake and Bazel source lists, test registration, options, and dependencies
   where both systems exist. For a Python package, check packaging and test
   discovery instead. Do not invent a second build system.
2. **Build and test.** Use [project-build](../project-build/SKILL.md). Run
   affected tests and the required broader checks. For non-trivial C++
   changes in repositories supporting both CMake and Bazel, validate both
   supported configurations and explain any platform/toolchain blocker.
3. **Lint and analysis.** Inspect `.lintrunner.toml`, `pyproject.toml`, and
   CI where present. Run configured checks for touched files; when using
   lintrunner, compare against this repository's actual base branch (which
   can be `master`, not `main`). Use the clang-tidy skill for applicable
   C++ changes. Scope automatic fixes and review their diff.
4. **Review.** Check names, API/exception contracts, ownership, portability,
   dependency boundaries, and test registration against local rules. Use
   available review tools when appropriate, without assuming a particular
   slash-command plugin exists. Report actual findings.
5. **Report.** State what passed, what failed, and what was not run and why.
   Identify known unrelated failures without silently fixing them or
   treating new failures as pre-existing.

For changes limited to these markdown rules and skills, validate frontmatter,
relative links, command references, and whitespace, then inspect the diff.
Compilation, runtime tests, and clang-tidy are unnecessary for that scope.
