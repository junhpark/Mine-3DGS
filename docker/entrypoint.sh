#!/usr/bin/env bash
# Thin entrypoint: everything goes through the CLI (§1.3 CLI 가 진실의 원천).
#   docker run --gpus all -v /path/to/data:/data minegs:gpu@sha256:... train run /data/dataset --profile light
set -euo pipefail
if [[ "${MINEGS_RUNTIME:-}" == "gpu" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[minegs] ERROR: GPU runtime image but no CUDA device visible (run with --gpus all)." >&2
    exit 2
  fi
fi
exec minegs "$@"
