#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从huggingface下载TravelUAV数据集的脚本，使用hfmirror加速下载
"""

import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='下载TravelUAV数据集')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/TravelUAV_data',
        help='数据集下载位置'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='wangxiangyu0814/TravelUAV_data_json',
        help='Huggingface上的数据集仓库ID'
    )
    parser.add_argument(
        '--mirror',
        type=str,
        default='https://hf-mirror.com',
        help='HuggingFace镜像站点URL'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='指定要下载的单个文件，不指定则下载整个仓库'
    )
    return parser.parse_args()

def download_dataset(args):
    """下载数据集"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始从 {args.repo_id} 下载数据集")
    logger.info(f"使用镜像: {args.mirror}")
    logger.info(f"下载位置: {output_dir.absolute()}")
    
    try:
        if args.file:
            # 下载单个文件
            file_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.file,
                repo_type="dataset",
                cache_dir=output_dir,
                endpoint=args.mirror
            )
            logger.info(f"文件已下载到: {file_path}")
        else:
            # 下载整个仓库
            downloaded_path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                endpoint=args.mirror
            )
            logger.info(f"数据集已下载到: {downloaded_path}")
        
        logger.info("下载完成!")
    except Exception as e:
        logger.error(f"下载过程中出错: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    download_dataset(args)