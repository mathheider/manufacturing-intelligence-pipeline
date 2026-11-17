#!/usr/bin/env bash
# One-shot reproduce: rebuild data, run the canonical benchmark, run the tests.
# No third-party packages required.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

echo "== 1/3  rebuild frozen data =="
python scripts/build_data.py

echo "== 2/3  canonical benchmark (14 comps, 40 seeds) =="
python -m mip.cli bench --seeds 40

echo "== 3/3  test suite (incl. golden-number regression) =="
python -m pytest -q
