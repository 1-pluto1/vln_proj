#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/../logs"
mkdir -p "${log_dir}"
ts="$(date '+%Y%m%d-%H%M%S')"
log_file="${log_dir}/server_${ts}.log"
exec > >(tee -a "${log_file}") 2>&1
echo "日志文件: ${log_file}"
PORT=25000
ROOT_PATH="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/TravelUAV_env_unzip"
SERVER_PY="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py"
PYTHON_BIN="python"
echo "启动命令: ${PYTHON_BIN} -u ${SERVER_PY} --port ${PORT} --root_path ${ROOT_PATH}"
${PYTHON_BIN} -u "${SERVER_PY}" --port "${PORT}" --root_path "${ROOT_PATH}"