#!/usr/bin/env bash
set -e

# Activate conda env if available
if [ -n "$CONDA_DEFAULT_ENV" ]; then
  : # already active
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate seqpred >/dev/null 2>&1 || true
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/opt/conda/etc/profile.d/conda.sh"
  conda activate seqpred >/dev/null 2>&1 || true
fi

cd "$(dirname "$0")/.." || exit 1

DATA_PATH="/SeqPred/data/data.csv"
TRAIN_START=0
TRAIN_END=4000
TEST_START=4000
TEST_END=5000
INITIAL_LENGTH=200
NUM_INTERVALS=3
AR_NOISE=0.0

EXP_ROOT="experiments/models"

for dir in "$EXP_ROOT"/*; do
  [ -d "$dir" ] || continue
  base="$(basename "$dir")"
  case "$base" in
    rnn_naive* ) mtype="rnn_naive" ;;
    lstm_naive* ) mtype="lstm_naive" ;;
    csept_smooth*|best_model_noise01 ) mtype="csept_smooth" ;;
    * ) echo "skip $base"; continue ;;
  esac
  echo "== Evaluating $base (type=$mtype) =="
  python test.py --model_type "$mtype" --model_path "/SeqPred/$dir/best_model" \
    --data_path "$DATA_PATH" \
    --train_start $TRAIN_START --train_end $TRAIN_END \
    --test_start $TEST_START --test_end $TEST_END \
    --initial_length $INITIAL_LENGTH \
    --num_intervals $NUM_INTERVALS \
    --autoregressive_noise_std $AR_NOISE \
    --plot \
    --output_dir "/SeqPred/$dir/evaluation"
  echo ""
done

# Generate comparison summary & plots
echo "== Generating comparison plots =="
python scripts/compare_experiments.py

