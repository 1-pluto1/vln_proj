from huggingface_hub import snapshot_download

import os
YOUR_HF_TOKEN = ""
# 可选：在代码内设置镜像与加速，避免每次终端手动导出
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 需安装 hf_transfer

# 指定下载位置
# cache_dir = "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpt/fis"
# 使用 HF Mirror 并开启并发与断点续传，仅下载核心文件
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    token=YOUR_HF_TOKEN,
    # cache_dir=cache_dir,
    endpoint="https://hf-mirror.com",  # 使用镜像源
    resume_download=True,              # 断点续传
    max_workers=8,                    # 并发连接数，按带宽调整
    allow_patterns=[
        "*.safetensors", "*.bin", "config.json", "tokenizer.json", "tokenizer.model", "*.pt"
    ],
    # proxies={"https": "http://your-proxy:port"}  # 如有需要启用代理
)


