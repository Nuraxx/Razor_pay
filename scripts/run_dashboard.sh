#!/usr/bin/env bash
# Launches the Adaptive Recovery dashboard.
#
#   ./scripts/run_dashboard.sh
#
# Runs fully offline (LLM_PROVIDER=mock by default -- see .env.example).
# No live Razorpay or LLM network calls are made.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./venv/bin/streamlit run ui/app.py
