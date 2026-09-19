# LinearAlgebra

Standalone C++ dense linear algebra library: Cholesky/LU/SVD decompositions,
linear solve, matrix inversion/multiplication/transpose, with a portable
scalar fallback and an optional Intel MKL backend. Source lives in
`include/`; tests live in `Testing/Cxx/`. Use namespace `linalg` and
the existing `LINALG_*` export/feature macros. Preserve the exception
reporting in `include/util/exception.h` (`LINALG_THROW`). Dependencies
are under `ThirdParty/`.

## Shared agent guidance

Adapted from the public [XSigma rules and skills](https://github.com/KhwarizmiAnalytix/XSigma/tree/89848c54492abef57fd0d0dc53b9da96b7cd1d5d)
at revision `89848c54492abef57fd0d0dc53b9da96b7cd1d5d`. Local API, dependency, language, and build
conventions below specialize that guidance for this standalone repository.

Read the applicable rules before editing. They apply to Claude as well as
Augment; C++ rules apply only when working on C++:

- [C++ coding](.augment/rules/coding.md) and [builders](.augment/rules/builder.md)
- [Python](.augment/rules/python.md)
- [Testing](.augment/rules/testing.md) and [builds](.augment/rules/build%20rule.md)
- [Dependencies](.augment/rules/ThirdParty.md)
- [Portability](.augment/rules/must-have.md) and [documentation](.augment/rules/markdown.md)

Use these task-specific skills as needed:

- [project-build](.claude/skills/project-build/SKILL.md): configure, build, and test
- [new-test](.claude/skills/new-test/SKILL.md): add tests using local conventions
- [clang-tidy](.claude/skills/clang-tidy/SKILL.md): analyze first-party C++ when applicable
- [session-checklist](.claude/skills/session-checklist/SKILL.md): verify completed work

## Build and test

Use the setup helper from `Scripts/`; inspect its help before adding
feature flags:

```sh
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test
```

For compiler or generator requirements, follow `README.md` and CI.
The repository also documents direct CMake commands for integration and CI.

For Bazel, also run from `Scripts/`:

```sh
python3 setup_bazel.py config.build.test
```

The scalar fallback paths are the default (`LINALG_ENABLE_MKL=OFF`); pass
`--linalg=MKL`-style flags (see `Scripts/setup.py --help`) or
`-DLINALG_ENABLE_MKL=ON` directly to exercise the MKL-backed paths.

## Test conventions

Follow neighboring Google Test cases and `LinearAlgebraTest.h`. Tests use
`Test*.cxx` under `Testing/Cxx/`; CMake uses a recursive glob while Bazel
uses a package-local glob. Check exclusions and register new subdirectories
in both systems. `Testing/Cxx/dense_matrix_test_helper.h` is a test-only
row-major dense matrix used to build fixtures — it is not part of the public
library and must not be included from `include/`.

## Verification and scope

For non-trivial source or build changes, run affected tests, review the diff,
and run configured lint/static-analysis checks relevant to touched files.
Check both build systems where provided. Follow the session checklist and
report checks run, failures, and unavailable tools explicitly. Guidance-only
changes need frontmatter/link/whitespace validation, not compilation.

Keep unrelated user edits and dependency sources intact. Share review
findings in the response or pull request; do not create unsolicited status
documents. Follow this repository's existing license and contribution policy.
