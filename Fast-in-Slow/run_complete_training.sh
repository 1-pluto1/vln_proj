#!/bin/bash

# FiS-UAV完整训练流程启动脚本
# 结合FiS异步采样和TravelUAV DAgger的完整训练系统

set -e  # 遇到错误时退出

# 默认参数
CONFIG_FILE=""
SAVE_DIR="./complete_training"
STAGE="all"
MODEL_V1=""
DEBUG=false
CREATE_CONFIG=false
GPU_ID=0
NUM_GPUS=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
FiS-UAV Complete Training Pipeline

Usage: $0 [OPTIONS]

Options:
    -c, --config FILE       Configuration file path
    -s, --save-dir DIR      Save directory (default: ./complete_training)
    -t, --stage STAGE       Training stage: 1, 2, or all (default: all)
    -m, --model-v1 PATH     Path to Model v1 (required for stage 2)
    -g, --gpu ID            GPU ID to use (default: 0)
    -n, --num-gpus NUM      Number of GPUs (default: 1)
    -d, --debug             Enable debug mode
    --create-config         Create default configuration file
    -h, --help              Show this help message

Stages:
    1    Initial training with TravelUAV 12k dataset (produces Model v1)
    2    DAgger closed-loop training (produces Model v2+)
    all  Run both stages sequentially

Examples:
    # Create default configuration
    $0 --create-config
    
    # Run complete training pipeline
    $0 --config fis_uav_complete_config.yaml
    
    # Run only initial training
    $0 --config config.yaml --stage 1
    
    # Run only DAgger training with existing Model v1
    $0 --config config.yaml --stage 2 --model-v1 ./models/model_v1.pt
    
    # Debug mode with specific GPU
    $0 --config config.yaml --debug --gpu 1

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -s|--save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        -t|--stage)
            STAGE="$2"
            shift 2
            ;;
        -m|--model-v1)
            MODEL_V1="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --create-config)
            CREATE_CONFIG=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 打印启动信息
print_info "🚁 FiS-UAV Complete Training Pipeline"
print_info "======================================"

# 创建默认配置
if [ "$CREATE_CONFIG" = true ]; then
    print_info "Creating default configuration file..."
    python run_complete_training.py --create_config
    print_success "Default configuration created: fis_uav_complete_config.yaml"
    print_info "Please edit the configuration file and run again with --config"
    exit 0
fi

# 检查配置文件
if [ -z "$CONFIG_FILE" ]; then
    print_error "Configuration file required. Use --create-config to create default config."
    show_help
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# 检查阶段2的模型路径
if [ "$STAGE" = "2" ] && [ -z "$MODEL_V1" ]; then
    print_error "Model v1 path required for stage 2. Use --model-v1 option."
    exit 1
fi

if [ "$STAGE" = "2" ] && [ ! -f "$MODEL_V1" ]; then
    print_error "Model v1 file not found: $MODEL_V1"
    exit 1
fi

# 环境检查
print_info "Checking environment..."

# 检查Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    print_info "CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    print_warning "CUDA not available, using CPU"
fi

# 检查PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    print_error "PyTorch not found. Please install PyTorch"
    exit 1
}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [ "$DEBUG" = true ]; then
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    print_info "Debug mode enabled"
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"
print_info "Save directory: $SAVE_DIR"

# 构建Python命令
PYTHON_CMD="python run_complete_training.py"
PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
PYTHON_CMD="$PYTHON_CMD --save_dir $SAVE_DIR"
PYTHON_CMD="$PYTHON_CMD --stage $STAGE"

if [ -n "$MODEL_V1" ]; then
    PYTHON_CMD="$PYTHON_CMD --model_v1 $MODEL_V1"
fi

if [ "$DEBUG" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --debug"
fi

# 打印训练配置
print_info "Training Configuration:"
print_info "  Config file: $CONFIG_FILE"
print_info "  Save directory: $SAVE_DIR"
print_info "  Stage: $STAGE"
print_info "  GPU ID: $GPU_ID"
print_info "  Number of GPUs: $NUM_GPUS"
if [ -n "$MODEL_V1" ]; then
    print_info "  Model v1: $MODEL_V1"
fi
print_info "  Debug mode: $DEBUG"

# 确认开始训练
echo
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training cancelled by user"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
print_success "Starting training at $(date)"

# 创建日志文件
LOG_FILE="$SAVE_DIR/training.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 运行训练
print_info "Executing: $PYTHON_CMD"
echo

# 使用tee同时输出到控制台和日志文件
if $PYTHON_CMD 2>&1 | tee "$LOG_FILE"; then
    # 计算训练时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo
    print_success "🎉 Training completed successfully!"
    print_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    print_success "Results saved to: $SAVE_DIR"
    print_success "Log file: $LOG_FILE"
    
    # 显示结果摘要
    if [ -f "$SAVE_DIR/complete_training_results.yaml" ]; then
        print_info "Training Summary:"
        python -c "
import yaml
try:
    with open('$SAVE_DIR/complete_training_results.yaml', 'r') as f:
        results = yaml.safe_load(f)
    
    print(f'  Pipeline Status: {results.get(\"pipeline_status\", \"unknown\")}')
    print(f'  Total Time: {results.get(\"total_training_time\", 0):.2f}s')
    
    if 'stage2' in results and 'dagger_results' in results['stage2']:
        dagger = results['stage2']['dagger_results']
        print(f'  DAgger Iterations: {dagger.get(\"total_iterations\", 0)}')
        print(f'  Final Success Rate: {dagger.get(\"final_success_rate\", 0):.3f}')
except Exception as e:
    print(f'  Could not parse results: {e}')
"
    fi
    
else
    # 训练失败
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo
    print_error "❌ Training failed after ${DURATION}s"
    print_error "Check log file for details: $LOG_FILE"
    
    # 显示最后几行日志
    if [ -f "$LOG_FILE" ]; then
        print_info "Last 10 lines of log:"
        tail -10 "$LOG_FILE"
    fi
    
    exit 1
fi