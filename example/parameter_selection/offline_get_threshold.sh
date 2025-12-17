#!/usr/bin/env bash
set -euo pipefail

# 最大阈值（包含自身），根据需要修改
MAX_THRESHOLD=10

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TS="$(date +%F_%H-%M-%S)"
OUTPUT_ROOT="${REPO_ROOT}/example/parameter_selection/threshold_output/${RUN_TS}"
RUN_OUTPUT_DIR="${REPO_ROOT}/offline_inference_output"

mkdir -p "${OUTPUT_ROOT}"

for threshold in $(seq 0 "${MAX_THRESHOLD}"); do
  echo ">>> Running threshold=${threshold}"
  python "${REPO_ROOT}/example/parameter_selection/offline_get_threshold.py" --min_chunk_threshold "${threshold}"

  # 找到最新生成的离线输出目录
  latest_dir="$(ls -1dt "${RUN_OUTPUT_DIR}"/*/ 2>/dev/null | head -n 1)"
  if [[ -z "${latest_dir}" ]]; then
    echo "!!! 未找到输出目录，跳过 threshold=${threshold}"
    continue
  fi

  src_plot_dir="${latest_dir%/}/replica_0/plots"
  dest_dir="${OUTPUT_ROOT}/${threshold}"
  mkdir -p "${dest_dir}"

  files=(
    "request_execution_time.csv"
    "request_execution_time_normalized.csv"
    "prefill_e2e_time.csv"
    "prefill_time_execution_plus_preemption_normalized.csv"
    "decode_time_execution_plus_preemption_normalized.csv"
  )

  for f in "${files[@]}"; do
    if [[ -f "${src_plot_dir}/${f}" ]]; then
      cp "${src_plot_dir}/${f}" "${dest_dir}/"
    else
      echo "!!! 未找到文件: ${src_plot_dir}/${f} (threshold=${threshold})"
    fi
  done

  echo ">>> Done threshold=${threshold}, files copied to ${dest_dir}"
done

echo "所有阈值运行完成，输出目录: ${OUTPUT_ROOT}"
