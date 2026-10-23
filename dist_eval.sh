#!/usr/bin/env bash
# Thin torchrun wrapper for GasVLM distributed evaluation.
# Usage: bash dist_eval.sh <script.py> <nproc_per_node> [extra args...]
#
# Distributed primitives (_init_dist_pytorch, collect_results_cpu, ...) are
# imported from gasvlm.utils.dist by the Python script — no PYTHONPATH hackery
# needed here. Run from the repo root after `pip install -e .`.

set -x

FILE=$1
NPROC=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((18500 + $RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Prefer `torchrun` if present; otherwise fall back to `python -m torch.distributed.run`.
if command -v torchrun &> /dev/null; then
    LAUNCHER=(torchrun)
else
    LAUNCHER=(python -m torch.distributed.run)
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${LAUNCHER[@]}" --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${PORT} \
  --nproc_per_node=${NPROC} \
  ${FILE} --launcher pytorch "${@:3}"
