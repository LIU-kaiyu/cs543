#!/usr/bin/env bash
# Run MiDaS over KITTI-C and compute final metrics.
# On Windows/PowerShell, call scripts/run_kittic_batch.py directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/run_kittic_batch.py" "$@"
