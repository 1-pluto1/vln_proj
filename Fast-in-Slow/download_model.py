from huggingface_hub import snapshot_download

# 指定下载位置
cache_dir = "/home/liusq/TravelUAV/vln_proj/Fast-in-Slow/ckpt"
# 使用 HF Mirror
snapshot_download(
    repo_id="haosad/fisvla", 
    cache_dir=cache_dir,
    endpoint="https://hf-mirror.com"  # 使用 HF Mirror 站点
)