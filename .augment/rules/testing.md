---
type: "always_apply"
---

# Tests and coverage

Cover changed behavior with focused tests for success, boundaries, invalid
inputs, failure handling, and relevant state changes. Keep tests independent
and deterministic; use fixtures when they reduce repeated setup.

Match the language, test framework, macros, filenames, and directory layout
already used by this repository. Consult `CLAUDE.md` and the `new-test`
skill. Do not import another project's test macros or helper headers.
Preserve existing exception assertions when exceptions are the public API.

Check how tests are registered. Add new files to explicit source lists and
verify glob patterns actually include them. Where CMake and Bazel both
exist, check both; avoid assuming their source lists are identical.

Run affected tests and the repository's configured coverage checks when
relevant. Report measured coverage honestly; do not assert a fixed 98%
merge gate unless the repository actually enforces it.
