#!/usr/bin/env bash
# Run the whole World Models pipeline, one stage after another.
#   ./run_all.sh            # laptop preset
#   ./run_all.sh paper      # paper preset (10,000 rollouts, CMA-ES pop 64 x 16 episodes)
#   ./run_all.sh laptop 3   # resume from step 3 (reuses the data / checkpoints on disk)
# Each stage can also be run on its own - see README.md.
set -euo pipefail
cd "$(dirname "$0")"
PRESET="${1:-laptop}"
FROM="${2:-1}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p runs

(( FROM <= 1 )) && { echo "[1/6] collect rollouts";   $PY scripts/01_collect_data.py   --preset "$PRESET" 2>&1 | tee runs/01_collect.log; }
(( FROM <= 2 )) && { echo "[2/6] train VAE";          $PY scripts/02_train_vae.py      --preset "$PRESET" 2>&1 | tee runs/02_vae.log; }
(( FROM <= 3 )) && { echo "[3/6] encode dataset";     $PY scripts/03_encode_data.py                       2>&1 | tee runs/03_encode.log; }
(( FROM <= 4 )) && { echo "[4/6] train MDN-RNN";      $PY scripts/04_train_mdrnn.py    --preset "$PRESET" 2>&1 | tee runs/04_mdrnn.log; }
(( FROM <= 5 )) && { echo "[5/6] train controller";   $PY scripts/05_train_controller.py --preset "$PRESET" 2>&1 | tee runs/05_controller.log; }
(( FROM <= 6 )) && { echo "[6/6] evaluate";           $PY scripts/06_evaluate.py --episodes 100           2>&1 | tee runs/06_evaluate.log; }
$PY scripts/visualize.py vae; $PY scripts/visualize.py dream; $PY scripts/visualize.py curves
$PY scripts/06_evaluate.py --episodes 1 --video runs/drive.mp4
