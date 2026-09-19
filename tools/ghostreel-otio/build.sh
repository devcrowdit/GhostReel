#!/usr/bin/env bash
# Build the ghostreel-otio sidecar into target/sidecars/.
# Thin wrapper: the real build lives in scripts/build-otio.mjs so Linux and Windows share it.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec node "${REPO_DIR}/scripts/build-otio.mjs" "$@"
