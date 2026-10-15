---
name: clang-tidy
description: Run or diagnose clang-tidy for first-party C++ under include/ and src/. Use for requested C++ static analysis or suppression fixes; this tool does not lint Python.
---

# clang-tidy

Read [CLAUDE.md](../../../CLAUDE.md), local `.clang-tidy` files where
present, and the build/linter configuration before choosing the invocation.

The setup helper supports a build-integrated clang-tidy pass. From
`Scripts/`, after checking toolchain prerequisites:

```sh
python3 setup.py config.build.ninja.clang.clangtidy
```

Read the repository's CMake clang-tidy integration before enabling fix mode;
fix flags can rewrite source during compilation.

`.lintrunner.toml` has a `CLANGTIDY` entry scoped to `include/`, `src/`,
and `Testing/`, with `ThirdParty/` excluded. It needs a `compile_commands.json`
from a build of this repo.

For an applicable C++ analysis:

- Use a clang-tidy version compatible with the compiler and standard library
  that produced `compile_commands.json`. Missing system headers often
  indicate a toolchain or database mismatch rather than a source defect.
- Confirm the compilation database includes each selected translation unit.
  Check headers through source files that include them; an absent database
  entry or skipped file is not a successful analysis.
- Scope fixes to the requested first-party code. Preserve unrelated working
  changes and inspect the diff after any automatic rewrite.
- Investigate a finding before suppressing it. Prefer a narrowly scoped
  suppression with its reason; directory configurations should retain parent
  settings. Do not disable checks globally to silence one false positive.
- Leave vendored code unchanged. Report findings separately from setup
  failures and state the actual files/configurations analyzed.
