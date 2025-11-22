import torch
print(f'PyTorch 编译支持的 CUDA 版本: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'PyTorch 运行时检测到的 CUDA 版本: {torch.version.cuda}')
    print(f'当前使用的 GPU 设备名称: {torch.cuda.get_device_name(0)}')
else:
    print('未检测到可用的 CUDA 设备，PyTorch 将运行在 CPU 模式。')
# 运行完毕后，输入 exit() 退出