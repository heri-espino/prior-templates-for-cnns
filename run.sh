#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash studies/cnn_release_experiment/run.sh "$@"
