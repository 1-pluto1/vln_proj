#!/usr/bin/env bash
set -euo pipefail

# ========= 配置默认值 =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
DATASETS_ROOT="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets"

HF_REPO_ID="wangxiangyu0814/UAV-Flow"
HF_MIRROR="https://hf-mirror.com"
RLDS_DATASET_NAME="uavflow_dataset"     # 写入到 RLDS JSON 的 dataset_name 字段
TFDS_DATASET_NAME="uav_dataset"          # TFDS Builder 名称，必须是 'uav_dataset'

# 路径约定
HF_OUT_DIR="${DATASETS_ROOT}/hf_out_data/datasets--wangxiangyu0814--UAV-Flow/snapshots/262d4cfa1ae05d578204583414c085c88ef2cfe7"
RLDS_DIR="${DATASETS_ROOT}/rlds_data"
TFDS_PARENT_DIR="${DATASETS_ROOT}"       # TFDS 构建的父目录（不含子目录名）
UAV_FOLDER_DIR="${DATASETS_ROOT}/uav_flow_data"  # 新增：Parquet 转换后的文件夹数据集输出

# 跳过选项
SKIP_DOWNLOAD=true
SKIP_CONVERT=true
SKIP_BUILD=false
HF_FILE=""            # 单文件（仓库内相对路径）
HF_FILES="train-00000-of-00054.parquet,train-00001-of-00054.parquet,train-00002-of-00054.parquet"           # 多文件，逗号分隔（仓库内相对路径）

# ========= 打印辅助 =========
red()    { echo -e "\033[31m$1\033[0m"; }
green()  { echo -e "\033[32m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }
blue()   { echo -e "\033[34m$1\033[0m"; }

usage() {
  cat <<EOF
用法: $0 [选项]

阶段:
  1) 从 HuggingFace 下载数据集 (dataset_build/dataset_download.py)
  1b) 转换 Parquet -> 文件夹数据集 (dataset_build/prepare_data.py 的同等逻辑)
  2) 转换为 RLDS JSON 结构 (dataset_build/uavflow2rlds.py)
  3) 构建 TFDS 数据集 (dataset_build/run_dataset_build.py)

选项:
  --hf-repo-id ID           HF 仓库ID (默认: ${HF_REPO_ID})
  --hf-mirror URL           HF 镜像URL (默认: ${HF_MIRROR})
  --datasets-root DIR       数据目录根路径 (默认: ${DATASETS_ROOT})
  --rlds-name NAME          RLDS 写入的 dataset_name (默认: ${RLDS_DATASET_NAME})
  --tfds-name NAME          TFDS Builder 名称 (默认: ${TFDS_DATASET_NAME})
  --tfds-output-dir DIR     TFDS 输出父目录（默认: ${TFDS_PARENT_DIR}）
  --hf-file FILE            仅下载仓库中的指定文件（单个）
  --hf-files F1,F2,...      仅下载仓库中的多个文件（逗号分隔）

  --skip-download           跳过下载
  --skip-convert            跳过 RLDS 转换
  --skip-build              跳过 TFDS 构建

示例:
  $0
  $0 --datasets-root /data/vln --hf-repo-id your/repo --rlds-name my_rlds --tfds-name uav_dataset
  $0 --skip-download --skip-convert   # 只进行 TFDS 构建
EOF
}

# ========= 解析参数 =========
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --hf-repo-id)    HF_REPO_ID="$2"; shift 2 ;;
    --hf-mirror)     HF_MIRROR="$2"; shift 2 ;;
    --datasets-root) DATASETS_ROOT="$2"; shift 2 ;;
    --rlds-name)     RLDS_DATASET_NAME="$2"; shift 2 ;;
    --tfds-name)     TFDS_DATASET_NAME="$2"; shift 2 ;;
    --tfds-output-dir) TFDS_PARENT_DIR="$2"; shift 2 ;;
    --hf-file)       HF_FILE="$2"; shift 2 ;;
    --hf-files)      HF_FILES="$2"; shift 2 ;;   # 新增：多文件
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --skip-convert)  SKIP_CONVERT=true; shift ;;
    --skip-build)    SKIP_BUILD=true; shift ;;
    *) red "未知参数: $1"; usage; exit 1 ;;
  esac
done

# ========= 显示配置 =========
blue "=== 构建配置 ==="
echo "ROOT_DIR:          ${ROOT_DIR}"
echo "DATASETS_ROOT:     ${DATASETS_ROOT}"
echo "HF_REPO_ID:        ${HF_REPO_ID}"
echo "HF_MIRROR:         ${HF_MIRROR}"
echo "HF_OUT_DIR:        ${HF_OUT_DIR}"
echo "UAV_FOLDER_DIR:    ${UAV_FOLDER_DIR}"
echo "RLDS_DIR:          ${RLDS_DIR}"
echo "TFDS_PARENT_DIR:   ${TFDS_PARENT_DIR}"   # 新增：展示 TFDS 输出父目录
echo "RLDS_DATASET_NAME: ${RLDS_DATASET_NAME}"
echo "TFDS_DATASET_NAME: ${TFDS_DATASET_NAME}"
echo

# ========= 检查依赖 =========
blue "=== 检查 Python 依赖 ==="
if ! command -v python3 >/dev/null 2>&1; then
  red "未找到 python3，请安装后重试"; exit 1
fi
if ! python3 -c "import tensorflow as tf; import tensorflow_datasets as tfds; import huggingface_hub; import datasets; import tqdm; import PIL" 2>/dev/null; then
  yellow "缺少依赖，请先安装：pip install tensorflow tensorflow-datasets huggingface_hub datasets tqdm pillow"
fi

# ========= 配置 PYTHONPATH =========
blue "=== 配置 Python 路径 ==="
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ========= 阶段 1: 下载 =========
if [[ "${SKIP_DOWNLOAD}" == "false" ]]; then
  blue "=== 阶段1: 从 HuggingFace 下载数据集 ==="
  mkdir -p "${HF_OUT_DIR}"

  if [[ -n "${HF_FILES}" ]]; then
    IFS=',' read -ra _FILES <<< "${HF_FILES}"
    for f in "${_FILES[@]}"; do
      echo "[下载] ${f}"
      python3 "${ROOT_DIR}/dataset_download.py" \
        --output_dir "${HF_OUT_DIR}" \
        --repo_id "${HF_REPO_ID}" \
        --mirror "${HF_MIRROR}" \
        --file "${f}"
    done
  elif [[ -n "${HF_FILE}" ]]; then
    python3 "${ROOT_DIR}/dataset_download.py" \
      --output_dir "${HF_OUT_DIR}" \
      --repo_id "${HF_REPO_ID}" \
      --mirror "${HF_MIRROR}" \
      --file "${HF_FILE}"
  else
    python3 "${ROOT_DIR}/dataset_download.py" \
      --output_dir "${HF_OUT_DIR}" \
      --repo_id "${HF_REPO_ID}" \
      --mirror "${HF_MIRROR}"
  fi

  green "✓ 下载完成 -> ${HF_OUT_DIR}"
else
  yellow "跳过下载阶段 (--skip-download)"
fi
echo

# ========= 阶段 1b: Parquet -> 文件夹数据集 =========
blue "=== 阶段1b: 转换 Parquet -> 文件夹数据集 ==="
mkdir -p "${UAV_FOLDER_DIR}"

# 选择扫描根：优先使用 --parquet-root，其次使用 HF_OUT_DIR
SCAN_ROOT="${PARQUET_ROOT:-${HF_OUT_DIR}}"
PARQUETS_TOTAL="$(find -L "${SCAN_ROOT}" -name "*.parquet" | wc -l)"
echo "在 ${SCAN_ROOT} 下发现 ${PARQUETS_TOTAL} 个 parquet 文件"

if [[ -n "${HF_FILES}" ]]; then
    IFS=',' read -ra _FILES <<< "${HF_FILES}"
    for f in "${_FILES[@]}"; do
        f_bn="$(basename "$f")"
        PARQUET_PATH="$(find -L "${SCAN_ROOT}" -name "${f_bn}" | head -n 1)"
        if [[ -z "${PARQUET_PATH}" ]]; then
            red "未在 ${SCAN_ROOT} 下找到指定文件: ${f}"
            echo "提示: 以下是前10个已发现的parquet文件："
            find -L "${SCAN_ROOT}" -name "*.parquet" | head -n 10
            exit 1
        fi
        echo "[转换] ${PARQUET_PATH} -> ${UAV_FOLDER_DIR}"
        python3 "${ROOT_DIR}/prepare_data.py" \
            --parquet_file "${PARQUET_PATH}" \
            --output_dir "${UAV_FOLDER_DIR}"
    done
elif [[ -n "${HF_FILE}" ]]; then
    f_bn="$(basename "$HF_FILE")"
    PARQUET_PATH="$(find -L "${SCAN_ROOT}" -name "${f_bn}" | head -n 1)"
    if [[ -z "${PARQUET_PATH}" ]]; then
        red "未在 ${SCAN_ROOT} 下找到指定文件: ${HF_FILE}"
        echo "提示: 以下是前10个已发现的parquet文件："
        find -L "${SCAN_ROOT}" -name "*.parquet" | head -n 10
        exit 1
    fi
    echo "[转换] ${PARQUET_PATH} -> ${UAV_FOLDER_DIR}"
    python3 "${ROOT_DIR}/prepare_data.py" \
        --parquet_file "${PARQUET_PATH}" \
        --output_dir "${UAV_FOLDER_DIR}"
else
    echo "[批量转换] 递归处理 ${SCAN_ROOT} 下所有 parquet -> ${UAV_FOLDER_DIR}"
    python3 "${ROOT_DIR}/prepare_data.py" \
        --parquet_root "${SCAN_ROOT}" \
        --output_dir "${UAV_FOLDER_DIR}"
fi

green "✓ 文件夹数据集生成 -> ${UAV_FOLDER_DIR}"
echo

# ========= 阶段 2: 转换为 RLDS =========
if [[ "${SKIP_CONVERT}" == "false" ]]; then
  blue "=== 阶段2: 转换为 RLDS JSON ==="
  if [[ ! -d "${UAV_FOLDER_DIR}" ]] || [[ -z "$(ls -A "${UAV_FOLDER_DIR}" 2>/dev/null)" ]]; then
    red "输入目录不存在或为空: ${UAV_FOLDER_DIR}"; exit 1
  fi
  mkdir -p "${RLDS_DIR}"
  python3 "${ROOT_DIR}/uavflow2rlds.py" \
    --input_dir "${UAV_FOLDER_DIR}" \
    --output_dir "${RLDS_DIR}" \
    --dataset_name "${RLDS_DATASET_NAME}"
  green "✓ RLDS 转换完成 -> ${RLDS_DIR}"
else
  yellow "跳过转换阶段 (--skip-convert)"
fi
echo

# ========= 阶段 3: 构建 TFDS =========
if [[ "${SKIP_BUILD}" == "false" ]]; then
  blue "=== 阶段3: 构建 TFDS 数据集 ==="
  if [[ ! -d "${RLDS_DIR}" ]] || [[ -z "$(ls -A "${RLDS_DIR}" 2>/dev/null)" ]]; then
    red "RLDS 目录不存在或为空: ${RLDS_DIR}"; exit 1
  fi
  # 确保 TFDS 输出父目录存在（支持自定义）
  mkdir -p "${TFDS_PARENT_DIR}"

  export RLDS_SOURCE_DIR="${RLDS_DIR}"      # 供 builder 读取源数据
  export TFDS_DATA_DIR="${TFDS_PARENT_DIR}" # 让 tfds.load 无需再传 data_dir

  python3 "${ROOT_DIR}/run_dataset_build.py" \
    --data_dir "${TFDS_PARENT_DIR}" \
    --dataset_name "${TFDS_DATASET_NAME}"

  BUILT_DIR="${TFDS_PARENT_DIR}/${TFDS_DATASET_NAME}/1.0.0"
  if [[ -d "${BUILT_DIR}" ]]; then
    green "✓ TFDS 构建完成 -> ${BUILT_DIR}"
    blue "=== 构建结果 ==="
    echo "TFDS 路径: ${BUILT_DIR}"
    echo "数据集大小: $(du -sh "${BUILT_DIR}" 2>/dev/null | cut -f1)"
    echo "轨迹数量: $(ls -1 "${RLDS_DIR}"/trajectory_* 2>/dev/null | wc -l)"
  else
    red "未找到构建输出目录: ${BUILT_DIR}"
    exit 1
  fi
else
  yellow "跳过 TFDS 构建阶段 (--skip-build)"
fi
echo

green "=== 全流程完成 ==="
echo "RLDS 源:   ${RLDS_DIR}"
echo "TFDS 父目录: ${TFDS_PARENT_DIR}"
echo "构建时间: $(date)"

unset RLDS_SOURCE_DIR || true
unset TFDS_DATA_DIR || true