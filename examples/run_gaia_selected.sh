#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/v3_bus.py"
SPLIT="validation"
LEVEL="all"
MAX_CONCURRENCY="1"
MAX_ROUNDS="20"
WORKDIR="workdir/v3_bus"
INDICES_CSV=""

usage() {
  cat <<'EOF'
Usage:
  bash examples/run_gaia_selected.sh --indices 7,9,16,22,24,30,34,80,88,132

Options:
  --indices           Comma-separated GAIA dataset indices to run.
  --config            Config file path. Default: configs/v3_bus.py
  --split             GAIA split. Default: validation
  --level             GAIA level selector. Default: all
  --max-concurrency   Passed through to run_gaia.py. Default: 1
  --max-rounds        Passed through to run_gaia.py. Default: 20
  --workdir           Workdir containing results/gaia. Default: workdir/v3_bus

Examples:
  bash examples/run_gaia_selected.sh --indices 7,9,16,22,24,30,34,80,88,132
  bash examples/run_gaia_selected.sh --indices 30,44,79 --level all --max-rounds 10
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --indices)
      INDICES_CSV="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --split)
      SPLIT="${2:-}"
      shift 2
      ;;
    --level)
      LEVEL="${2:-}"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="${2:-}"
      shift 2
      ;;
    --max-rounds)
      MAX_ROUNDS="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$INDICES_CSV" ]]; then
  echo "--indices is required" >&2
  usage >&2
  exit 1
fi

RESULTS_DIR="$ROOT_DIR/$WORKDIR/results/gaia"
mkdir -p "$RESULTS_DIR"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
AGG_JSON="$RESULTS_DIR/benchmark_gaia_selected_${TIMESTAMP}.json"

IFS=',' read -r -a RAW_INDICES <<< "$INDICES_CSV"
INDICES=()
for raw in "${RAW_INDICES[@]}"; do
  trimmed="$(echo "$raw" | xargs)"
  [[ -z "$trimmed" ]] && continue
  if ! [[ "$trimmed" =~ ^[0-9]+$ ]]; then
    echo "Invalid index: $trimmed" >&2
    exit 1
  fi
  INDICES+=("$trimmed")
done

if [[ "${#INDICES[@]}" -eq 0 ]]; then
  echo "No valid indices parsed from --indices" >&2
  exit 1
fi

echo "Running selected GAIA tasks: ${INDICES[*]}"
echo "Config: $CONFIG"
echo "Results dir: $RESULTS_DIR"
echo

run_count=0
for idx in "${INDICES[@]}"; do
  echo "=================================================="
  echo "[$((run_count + 1))/${#INDICES[@]}] Running GAIA index $idx"
  echo "=================================================="

  before_latest="$(ls -1t "$RESULTS_DIR"/benchmark_gaia_*.json 2>/dev/null | head -n 1 || true)"

  python examples/run_gaia.py \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --level "$LEVEL" \
    --start "$idx" \
    --end "$((idx + 1))" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --max-rounds "$MAX_ROUNDS"

  after_latest="$(ls -1t "$RESULTS_DIR"/benchmark_gaia_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "$after_latest" ]]; then
    echo "Failed to locate result file after running index $idx" >&2
    exit 1
  fi
  if [[ "$after_latest" == "$before_latest" ]]; then
    echo "Warning: latest result file did not change for index $idx, still using $after_latest" >&2
  fi

  python - "$after_latest" "$idx" "$TMP_DIR" <<'PY'
import json
import os
import sys

result_file, index_str, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
with open(result_file, encoding="utf-8") as f:
    data = json.load(f)
results = data.get("results", [])
if len(results) != 1:
    raise SystemExit(f"Expected exactly 1 result in {result_file}, got {len(results)}")
record = dict(results[0])
record["selected_index"] = int(index_str)
record["source_file"] = os.path.basename(result_file)
out_path = os.path.join(out_dir, f"{int(index_str):04d}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
PY

  run_count=$((run_count + 1))
  echo
done

python - "$TMP_DIR" "$AGG_JSON" "$CONFIG" "$SPLIT" "$LEVEL" "$MAX_CONCURRENCY" "$MAX_ROUNDS" "$INDICES_CSV" <<'PY'
import json
import os
import sys
from datetime import datetime

tmp_dir, agg_json, config, split, level, max_concurrency, max_rounds, indices_csv = sys.argv[1:]

records = []
for name in sorted(os.listdir(tmp_dir)):
    if name.endswith(".json"):
        with open(os.path.join(tmp_dir, name), encoding="utf-8") as f:
            records.append(json.load(f))

scored = [r for r in records if r.get("correct") is not None]
correct = sum(1 for r in scored if r.get("correct") is True)
wrong = sum(1 for r in scored if r.get("correct") is False)
unscored = len(records) - len(scored)
avg_time = sum(float(r.get("processing_time", 0.0) or 0.0) for r in records) / len(records) if records else 0.0

per_level_total = {}
per_level_correct = {}
for record in scored:
    level_key = str(record.get("level", "") or "")
    per_level_total[level_key] = per_level_total.get(level_key, 0) + 1
    if record.get("correct") is True:
        per_level_correct[level_key] = per_level_correct.get(level_key, 0) + 1

per_level_accuracy = {
    key: (per_level_correct.get(key, 0) / total if total else 0.0)
    for key, total in sorted(per_level_total.items())
}

payload = {
    "experiment_meta": {
        "timestamp": datetime.now().isoformat() + "Z",
        "benchmark": "gaia_selected",
        "config": config,
        "split": split,
        "level": level,
        "concurrency": int(max_concurrency),
        "max_rounds": int(max_rounds),
        "selected_indices": [int(x.strip()) for x in indices_csv.split(",") if x.strip()],
        "total_tasks": len(records),
    },
    "results": records,
    "summary": {
        "completed_tasks": len(records),
        "scored_tasks": len(scored),
        "unscored_tasks": unscored,
        "correct_answers": correct,
        "wrong_answers": wrong,
        "accuracy": (correct / len(scored)) if scored else None,
        "average_time": avg_time,
        "per_level_accuracy": per_level_accuracy,
        "last_updated": datetime.now().isoformat() + "Z",
    },
}

with open(agg_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print("==================================================")
print("Selected GAIA Summary")
print("==================================================")
print(f"Tasks run:      {len(records)}")
print(f"Scored tasks:   {len(scored)}")
print(f"Unscored tasks: {unscored}")
if scored:
    print(f"Correct:        {correct}")
    print(f"Wrong:          {wrong}")
    print(f"Accuracy:       {correct / len(scored):.2%}")
else:
    print("Accuracy:       N/A")
print(f"Average time:   {avg_time:.2f}s")
print(f"Per-level acc:  {per_level_accuracy}")
print(f"Saved to:       {agg_json}")
PY
