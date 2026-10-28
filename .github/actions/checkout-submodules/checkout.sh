#!/usr/bin/env bash
set -euo pipefail

# Keep submodule downloads separate from actions/checkout so hosting failures
# can recover without discarding the main checkout or completed downloads.
max_attempts=5
retry_delay=15
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "Fetching recursive submodules (attempt $attempt/$max_attempts)"
    if git submodule sync --recursive &&
        git -c protocol.version=2 submodule update --init --force --depth=1 --recursive; then
        exit 0
    fi

    if ((attempt == max_attempts)); then
        echo "::error::Submodule checkout failed after $max_attempts attempts."
        exit 1
    fi

    echo "::warning::Submodule checkout failed; retrying in $retry_delay seconds."
    sleep "$retry_delay"
    retry_delay=$((retry_delay * 2))
done
