#!/usr/bin/env bash
# Usage: bash scripts/build_sglang.sh 12.9 [--prepare | --check-deps]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/tmp/sglang_build_cu129.log}"
"${PYTHON:-python3}" -u "$SCRIPT_DIR/build_sglang.py" "$@" 2>&1 | tee -a "$LOG_FILE"
