# 测试 PyTorch 和 YOLOv8 是否安装成功

import torch

print("="*50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch 使用的 CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
else:
    print("❌ GPU 不可用原因:")
    print(f"- CUDA Toolkit 安装: {'已安装' if hasattr(torch.version, 'cuda') else '未安装'}")
    print(f"- 驱动版本: {torch.cuda._get_nvml_driver_version() if hasattr(torch.cuda, '_get_nvml_driver_version') else '无法获取'}")
print("="*50)