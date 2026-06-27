#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_long_matrix.sh [experiment] [model] [dataset] [gpu] [dry_run]
#
# Positional arguments accept "all" or comma-separated values.
#   experiment: all | numeric | mm_tsflib | mm_tsflib_nonlinear | mm_tsflib_nonlinear_bounded | mm_tsflib_nonlinear_shrink | mm_tsflib_nonlinear_shrink_signed | mm_tsflib_vot_freq | mm_tsflib_freq_residual | mm_tsflib_vot_freq_shrink | mm_tsflib_vot_freq_shrink_signed | llm_generated | random_text
#   model:      all | DLinear | PatchTST
#   dataset:    all | Energy | Public_Health | ...
#   gpu:        CUDA_VISIBLE_DEVICES id, default 0
#   dry_run:    1 prints commands only, default 0
#
# Examples:
#   bash scripts/run_long_matrix.sh
#   bash scripts/run_long_matrix.sh numeric PatchTST Public_Health 0
#   bash scripts/run_long_matrix.sh mm_tsflib PatchTST Energy,Public_Health 0 1
#   bash scripts/run_long_matrix.sh mm_tsflib_nonlinear PatchTST Energy,Public_Health 0 1
#   bash scripts/run_long_matrix.sh mm_tsflib_vot_freq PatchTST Energy,Public_Health 0 1
#   bash scripts/run_long_matrix.sh mm_tsflib_vot_freq_shrink PatchTST Energy,Public_Health 0 1
#
# In the submit package, llm_generated reads from:
#   data/llm-generated/<Dataset>_H<SEQ_LEN>_F<pred_len>_ecnu_llm.csv
# while keeping root_path as data/<Dataset> so result parsers preserve the
# original dataset name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

EXPERIMENT_ARG="${1:-all}"
MODEL_ARG="${2:-all}"
DATASET_ARG="${3:-all}"
export CUDA_VISIBLE_DEVICES="${4:-0}"
DRY_RUN="${5:-0}"

ALL_EXPERIMENTS=(
  "numeric"
  "mm_tsflib"
  "mm_tsflib_nonlinear"
  "mm_tsflib_nonlinear_bounded"
  "mm_tsflib_nonlinear_shrink"
  "mm_tsflib_nonlinear_shrink_signed"
  "mm_tsflib_vot_freq"
  "mm_tsflib_freq_residual"
  "mm_tsflib_vot_freq_shrink"
  "mm_tsflib_vot_freq_shrink_signed"
  "llm_generated"
  "random_text"
)
ALL_MODELS=("DLinear" "PatchTST")
ALL_DATASETS=("Algriculture" "Climate" "Economy" "Energy" "Public_Health" "Security" "SocialGood" "Traffic")

PRED_LENS="${PRED_LENS:-12,24,36,48}"
SEEDS="${SEEDS:-2021}"
SEQ_LEN="${SEQ_LEN:-24}"
LABEL_LEN="${LABEL_LEN:-12}"
TEXT_LEN="${TEXT_LEN:-4}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
E_LAYERS="${E_LAYERS:-2}"
D_FF="${D_FF:-256}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-0.0001}"
LLM_MODEL="${LLM_MODEL:-BERT}"
PROMPT_WEIGHT="${PROMPT_WEIGHT:-0.1}"
VOT_LOW_FREQ_RATIO="${VOT_LOW_FREQ_RATIO:-0.1}"
VOT_HIGH_FREQ_RATIO="${VOT_HIGH_FREQ_RATIO:-0.3}"
FUSION_BAND_DELTA_MAX="${FUSION_BAND_DELTA_MAX:-0.05}"
FUSION_GATE_DELTA_MAX="${FUSION_GATE_DELTA_MAX:-0.03}"
FUSION_SHRINK_INIT="${FUSION_SHRINK_INIT:-0.1}"
FUSION_SHRINK_MAX="${FUSION_SHRINK_MAX:-0.5}"
FUSION_SHRINK_SIGNED="${FUSION_SHRINK_SIGNED:-0}"
POOL_TYPE="${POOL_TYPE:-avg}"
USE_FULLMODEL="${USE_FULLMODEL:-0}"
RANDOM_TEXT_COLUMN="${RANDOM_TEXT_COLUMN:-Random_Text}"
RANDOM_SOURCE_TEXT_COLUMN="${RANDOM_SOURCE_TEXT_COLUMN:-Final_Search_${TEXT_LEN}}"
RANDOM_TEXT_SEED="${RANDOM_TEXT_SEED:-20240624}"
RANDOM_TEXT_MODE="${RANDOM_TEXT_MODE:-char_noise}"
BUILD_RANDOM_TEXT="${BUILD_RANDOM_TEXT:-1}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./results}"

select_items() {
  local selected="$1"
  shift
  if [[ "${selected}" == "all" ]]; then
    printf '%s\n' "$@"
  else
    tr ',' '\n' <<< "${selected}"
  fi
}

data_path_for_dataset() {
  case "$1" in
    Algriculture) echo "US_RetailBroilerComposite_Month.csv" ;;
    Climate) echo "US_precipitation_month.csv" ;;
    Economy) echo "US_TradeBalance_Month.csv" ;;
    Energy) echo "US_GasolinePrice_Week.csv" ;;
    Public_Health) echo "US_FLURATIO_Week.csv" ;;
    Security) echo "US_FEMAGrant_Month.csv" ;;
    SocialGood) echo "Unadj_UnemploymentRate_ALL_processed.csv" ;;
    Traffic) echo "US_VMT_Month.csv" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
}

valid_combo() {
  local csv_path="$1"
  local pred_len="$2"
  local dataset="$3"

  if [[ ! -f "${csv_path}" ]]; then
    echo "Missing data file: ${csv_path}" >&2
    return 1
  fi

  local n_lines n_rows num_train num_test num_vali train_windows val_windows test_windows
  n_lines="$(wc -l < "${csv_path}")"
  n_rows=$((n_lines - 1))
  num_train=$((n_rows * 70 / 100))
  num_test=$((n_rows * 20 / 100))
  num_vali=$((n_rows - num_train - num_test))

  train_windows=$((num_train - SEQ_LEN - pred_len + 1))
  val_windows=$((num_vali - pred_len + 1))
  test_windows=$((num_test - pred_len + 1))

  if (( train_windows <= 0 || val_windows <= 0 || test_windows <= 0 )); then
    echo "Skipping ${dataset} pred_len=${pred_len}: insufficient split length (train=${train_windows}, val=${val_windows}, test=${test_windows})"
    return 1
  fi
  return 0
}

metric_is_finite() {
  local metric_path="$1"
  python - "${metric_path}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

for key in ("mse", "mae"):
    value = data.get(key)
    if value is None or not math.isfinite(float(value)):
        sys.exit(1)
sys.exit(0)
PY
}

mapfile -t EXPERIMENTS < <(select_items "${EXPERIMENT_ARG}" "${ALL_EXPERIMENTS[@]}")
mapfile -t MODELS < <(select_items "${MODEL_ARG}" "${ALL_MODELS[@]}")
mapfile -t DATASETS < <(select_items "${DATASET_ARG}" "${ALL_DATASETS[@]}")
mapfile -t PRED_LEN_LIST < <(tr ',' '\n' <<< "${PRED_LENS}")
mapfile -t SEED_LIST < <(tr ',' '\n' <<< "${SEEDS}")

mkdir -p "${OUTPUT_BASE_DIR}"

for experiment in "${EXPERIMENTS[@]}"; do
  for model_name in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      root_path="./data/${dataset}"
      raw_data_path="$(data_path_for_dataset "${dataset}")"
      data_path="${raw_data_path}"
      if [[ "${experiment}" == "random_text" ]]; then
        data_path="${raw_data_path%.csv}_random_text.csv"
        if [[ ! -f "${root_path}/${data_path}" && "${BUILD_RANDOM_TEXT}" == "1" ]]; then
          build_cmd=(
            python scripts/build_random_text_control_dataset.py
            --input "${root_path}/${raw_data_path}"
            --output "${root_path}/${data_path}"
            --source-text-column "${RANDOM_SOURCE_TEXT_COLUMN}"
            --text-column "${RANDOM_TEXT_COLUMN}"
            --mode "${RANDOM_TEXT_MODE}"
            --seed "${RANDOM_TEXT_SEED}"
          )
          printf 'Build random text control:'
          printf ' %q' "${build_cmd[@]}"
          printf '\n'
          if [[ "${DRY_RUN}" != "1" ]]; then
            "${build_cmd[@]}"
          fi
        fi
      fi

      for seed in "${SEED_LIST[@]}"; do
        for pred_len in "${PRED_LEN_LIST[@]}"; do
          data_path="${raw_data_path}"
          if [[ "${experiment}" == "llm_generated" ]]; then
            # Keep root_path at ./data/<dataset> so downstream result parsers
            # still recover the original dataset name from args.root_path.
            # The actual file is read from submit/data/llm-generated/.
            data_path="../llm-generated/${dataset}_H${SEQ_LEN}_F${pred_len}_ecnu_llm.csv"
          elif [[ "${experiment}" == "random_text" ]]; then
            data_path="${raw_data_path%.csv}_random_text.csv"
          fi

          model_id="${dataset}_${model_name}_${experiment}_s${seed}_sl${SEQ_LEN}_pl${pred_len}"
          output_dir="${OUTPUT_BASE_DIR}/${experiment}_${model_name}_long_term"
          save_name="${OUTPUT_BASE_DIR}/${experiment}_${model_name}_summary.txt"
          combo_check_path="${root_path}/${data_path}"
          if [[ "${DRY_RUN}" == "1" && "${experiment}" == "random_text" && ! -f "${combo_check_path}" ]]; then
            combo_check_path="${root_path}/${raw_data_path}"
          fi

          if ! valid_combo "${combo_check_path}" "${pred_len}" "${dataset}"; then
            continue
          fi

          existing_metric="$(compgen -G "${output_dir}/long_term_forecast_${model_id}_*/metrics.json" | head -n 1 || true)"
          if [[ -n "${existing_metric}" ]]; then
            if metric_is_finite "${existing_metric}"; then
              echo "Skipping existing finite result: ${model_id}"
              continue
            fi
            echo "Re-running invalid existing result: ${model_id} (${existing_metric})"
          fi

          cmd=(
            python -u run.py
            --task_name long_term_forecast
            --experiment "${experiment}"
            --is_training 1
            --root_path "${root_path}"
            --data_path "${data_path}"
            --model_id "${model_id}"
            --model "${model_name}"
            --data custom
            --features S
            --seq_len "${SEQ_LEN}"
            --label_len "${LABEL_LEN}"
            --pred_len "${pred_len}"
            --des "${experiment}_matrix"
            --seed "${seed}"
            --text_len "${TEXT_LEN}"
            --train_epochs "${EPOCHS}"
            --patience "${PATIENCE}"
            --batch_size "${BATCH_SIZE}"
            --num_workers "${NUM_WORKERS}"
            --d_model "${D_MODEL}"
            --n_heads "${N_HEADS}"
            --e_layers "${E_LAYERS}"
            --d_ff "${D_FF}"
            --dropout "${DROPOUT}"
            --learning_rate "${LR}"
            --save_name "${save_name}"
            --output_dir "${output_dir}"
          )

          if [[ "${experiment}" == "mm_tsflib" || "${experiment}" == "mm_tsflib_nonlinear" || "${experiment}" == "mm_tsflib_nonlinear_bounded" || "${experiment}" == "mm_tsflib_nonlinear_shrink" || "${experiment}" == "mm_tsflib_nonlinear_shrink_signed" || "${experiment}" == "mm_tsflib_vot_freq" || "${experiment}" == "mm_tsflib_freq_residual" || "${experiment}" == "mm_tsflib_vot_freq_shrink" || "${experiment}" == "mm_tsflib_vot_freq_shrink_signed" || "${experiment}" == "llm_generated" || "${experiment}" == "random_text" ]]; then
            cmd+=(
              --type_tag "#F#"
              --prompt_weight "${PROMPT_WEIGHT}"
              --pool_type "${POOL_TYPE}"
              --llm_model "${LLM_MODEL}"
              --huggingface_token "NA"
              --use_fullmodel "${USE_FULLMODEL}"
            )
          fi

          if [[ "${experiment}" == "mm_tsflib_vot_freq" || "${experiment}" == "mm_tsflib_freq_residual" || "${experiment}" == "mm_tsflib_vot_freq_shrink" || "${experiment}" == "mm_tsflib_vot_freq_shrink_signed" ]]; then
            cmd+=(
              --vot_low_freq_ratio "${VOT_LOW_FREQ_RATIO}"
              --vot_high_freq_ratio "${VOT_HIGH_FREQ_RATIO}"
            )
          fi

          if [[ "${experiment}" == "mm_tsflib_freq_residual" ]]; then
            cmd+=(--fusion_band_delta_max "${FUSION_BAND_DELTA_MAX}")
          fi

          if [[ "${experiment}" == "mm_tsflib_nonlinear_bounded" ]]; then
            cmd+=(--fusion_gate_delta_max "${FUSION_GATE_DELTA_MAX}")
          fi

          if [[ "${experiment}" == "mm_tsflib_nonlinear_shrink" || "${experiment}" == "mm_tsflib_nonlinear_shrink_signed" || "${experiment}" == "mm_tsflib_vot_freq_shrink" || "${experiment}" == "mm_tsflib_vot_freq_shrink_signed" ]]; then
            cmd+=(
              --fusion_residual_shrink 1
              --fusion_shrink_init "${FUSION_SHRINK_INIT}"
              --fusion_shrink_max "${FUSION_SHRINK_MAX}"
            )
            if [[ "${FUSION_SHRINK_SIGNED}" == "1" || "${experiment}" == *"_shrink_signed" ]]; then
              cmd+=(--fusion_shrink_signed 1)
            fi
          fi

          if [[ "${experiment}" == "llm_generated" ]]; then
            cmd+=(
              --text_column "ECNU_LLM_Text"
              --text_origin_offset "-1"
              --prior_mode "origin_repeat"
            )
          elif [[ "${experiment}" == "random_text" ]]; then
            cmd+=(
              --text_column "${RANDOM_TEXT_COLUMN}"
              --text_origin_offset "-1"
              --prior_mode "origin_repeat"
            )
          elif [[ "${experiment}" == "mm_tsflib" || "${experiment}" == "mm_tsflib_nonlinear" || "${experiment}" == "mm_tsflib_nonlinear_bounded" || "${experiment}" == "mm_tsflib_nonlinear_shrink" || "${experiment}" == "mm_tsflib_nonlinear_shrink_signed" || "${experiment}" == "mm_tsflib_vot_freq" || "${experiment}" == "mm_tsflib_freq_residual" || "${experiment}" == "mm_tsflib_vot_freq_shrink" || "${experiment}" == "mm_tsflib_vot_freq_shrink_signed" ]]; then
            cmd+=(
              --text_origin_offset "-1"
              --prior_mode "origin_repeat"
            )
          fi

          echo "Running ${experiment} | ${model_name} | ${dataset} | pred_len=${pred_len} | seed=${seed}"
          printf 'Command:'
          printf ' %q' "${cmd[@]}"
          printf '\n'

          if [[ "${DRY_RUN}" != "1" ]]; then
            "${cmd[@]}"
          fi
        done
      done
    done
  done
done
