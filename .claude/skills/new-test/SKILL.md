---
name: new-test
description: Add or extend a unit test in Logging, preserving its framework, test registration, and error-handling contracts. Use when writing tests for changed behavior.
---

# new-test

Read [CLAUDE.md](../../../CLAUDE.md) and the existing test closest to
the changed behavior before choosing a filename, fixture, or macro.

Follow neighboring Google Test cases and `LoggingTest.h`. Tests use
`Test*.cpp` under `Testing/Cxx/`; CMake uses a recursive glob while Bazel
uses a package-local glob. Check exclusions and register new subdirectories
in both systems. Backend changes need coverage for the affected backend;
dispatch changes should exercise all four backends.

1. Search for existing coverage with `rg`; extend the relevant test file
   instead of creating a duplicate suite.
2. Match the neighboring includes, namespace, fixtures, naming, and license
   conventions. Do not bring in test helpers from a different repository.
3. Cover the meaningful success, boundary, and failure cases for the change.
   Assert public behavior and the repository's documented error contract.
4. Check source registration, discovery patterns, and backend exclusions so
   the new cases actually execute. Keep parallel build definitions in sync
   where both exist.
5. Run the affected tests using [project-build](../project-build/SKILL.md).
   Report the test command and result, including any unavailable backend.
