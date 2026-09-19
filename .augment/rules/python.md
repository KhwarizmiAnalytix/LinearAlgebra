---
type: "always_apply"
---

# Python coding standards

Apply these rules to Python files, including build scripts in C++ projects.

- Follow the Google Python style conventions used by nearby code. Use
  `snake_case` functions/modules, `CapWords` classes, and descriptive types
  and Google-style docstrings on public interfaces.
- Respect `requires-python` and existing tool target versions. Preserve
  Python 3.9 compatibility where that is the declared minimum.
- Prefer `pathlib`, context managers, portable subprocess argument lists,
  and standard-library platform abstractions.
- Group standard, third-party, and project imports using existing formatter
  settings. Use the repository's current formatter/linter; do not install
  competing Black, Ruff, pylint, or import-sorting configurations.
- Catch specific exceptions and preserve useful error context. The C++
  return-value preference does not ban idiomatic Python exceptions.
- Use the existing pytest or unittest conventions, deterministic fixtures,
  and temporary directories for file-system tests. Mock unavailable external
  tools at their boundary and test the resulting success/failure behavior.
- Keep credentials out of source and generated examples.
