#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# run.sh — launch the Streamlit demo with the environment required to keep
# Python 3.13 + Apple Silicon + torch + transformers + tornado happy together.
#
# Without these vars, sentence-transformers + streamlit segfaults at startup
# on this stack (observed 2026-05-07). Each variable disables a different
# threading layer; setting them in the shell BEFORE streamlit launches is
# important because OMP/MKL initialize their thread pools at C-library
# import time, which happens before python-dotenv has a chance to run.
#
# Usage:   ./run.sh
# ----------------------------------------------------------------------------

set -e
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1            # disable OpenMP parallelism
export MKL_NUM_THREADS=1            # disable Intel MKL parallelism
export TOKENIZERS_PARALLELISM=false # disable HF tokenizer parallelism
export JOBLIB_MULTIPROCESSING=0     # force joblib serial backend
export PYTORCH_ENABLE_MPS_FALLBACK=1 # let MPS fall back to CPU on unsupported ops

exec streamlit run app.py --server.fileWatcherType=none
