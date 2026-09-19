# Contributing to LinearAlgebra

## Building

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Enable the optional MKL backend with `-DLINALG_ENABLE_MKL=ON` and any other
`-DLINALG_ENABLE_*` flag documented in
[README.md](README.md#cmake-options). Bazel: `bazel test //...`.

## Before opening a PR

- Run the test suite (default: scalar fallback, `LINALG_ENABLE_MKL=OFF`); if
  your change touches an MKL-gated code path, also build with
  `-DLINALG_ENABLE_MKL=ON` if MKL is available to you.
- For changes touching memory or object lifetime, build with a sanitizer:
  ```bash
  cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DLINALG_ENABLE_SANITIZER=ON -DLINALG_SANITIZER_TYPE=address
  cmake --build build-asan && ctest --test-dir build-asan --output-on-failure
  ```
- Add or update a test under `Testing/Cxx/` for behavior changes.
- Keep public CMake/Bazel option changes documented in `README.md`.

## Coverage locally

```bash
cmake -S . -B build-cov -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLINALG_ENABLE_COVERAGE=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
lcov --capture --directory build-cov --output-file coverage.info --ignore-errors mismatch,negative,gcov,source
lcov --remove coverage.info '*/ThirdParty/*' '*/Testing/*' '/usr/*' --ignore-errors unused --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

## Lint

```bash
pip install lintrunner lint-tool
lintrunner init
lintrunner -a
lintrunner
```
