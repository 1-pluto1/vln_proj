#!/usr/bin/env python3
import torch
import os

print("=== GPU检测报告 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 列出所有GPU
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 测试GPU可用性
        try:
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(10, 10).to(device)
            print(f"  ✓ GPU {i} 可正常使用")
        except Exception as e:
            print(f"  ✗ GPU {i} 错误: {e}")
else:
    print("CUDA不可用!")

print("\n=== NCCL环境变量检查 ===")
nccl_vars = [
    'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME', 'NCCL_P2P_DISABLE', 
    'NCCL_IB_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_TIMEOUT'
]

for var in nccl_vars:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")