#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${SCRIPT_DIR}/.build-venv"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt" -r "${SCRIPT_DIR}/requirements-build.txt"

DIST_DIR="${REPO_DIR}/target/sidecars"
mkdir -p "${DIST_DIR}"

"${VENV_DIR}/bin/pyinstaller" \
    --onefile \
    --name ghostreel-otio \
    --collect-all opentimelineio \
    --collect-all otio_fcp_adapter \
    --copy-metadata otio-fcp-adapter \
    --distpath "${DIST_DIR}" \
    --workpath "${REPO_DIR}/target/sidecars-build" \
    --specpath "${REPO_DIR}/target/sidecars-build" \
    "${SCRIPT_DIR}/ghostreel_otio.py"

echo "Built ${DIST_DIR}/ghostreel-otio"
