#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face缓存配置模块
统一设置所有Hugging Face相关的缓存路径到自定义目录
"""

import os
import sys
import logging

def setup_hf_cache(custom_cache_dir: str = None, enable_mirror: bool = True, verbose: bool = True):
    """
    设置Hugging Face缓存路径到自定义目录
    
    Args:
        custom_cache_dir: 自定义缓存目录，None则自动使用项目根目录下的 .hf_cache
        enable_mirror: 是否启用 HF-Mirror 镜像站
        verbose: 是否打印详细信息
    
    Returns:
        str: 缓存目录路径
    """
    
    # 确定缓存目录
    if custom_cache_dir is None:
        custom_cache_dir = get_cache_dir()
    
    # 创建缓存目录
    os.makedirs(custom_cache_dir, exist_ok=True)
    
    # 设置所有Hugging Face相关的环境变量
    os.environ['HF_HOME'] = custom_cache_dir
    os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
    os.environ['HF_HUB_CACHE'] = custom_cache_dir
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    # 设置镜像站（可选）
    if enable_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_BASE_URL'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    # 设置模型和数据集缓存
    os.environ['HF_MODELS_CACHE'] = custom_cache_dir
    os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
    os.environ['HF_HUB_CACHE'] = custom_cache_dir
    
    # 设置镜像站相关配置
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    
    # 设置日志级别
    os.environ['HF_DATASETS_VERBOSITY'] = 'error'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_VERBOSITY'] = 'error'
    
    # 设置更强的网络超时和重试配置
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 增加到10分钟
    os.environ['HF_HUB_DOWNLOAD_RETRY_DELAY'] = '10'  # 增加重试延迟
    os.environ['HF_HUB_DOWNLOAD_MAX_RETRIES'] = '5'  # 增加重试次数
    
    # 设置requests库的超时配置
    os.environ['REQUESTS_TIMEOUT'] = '600'
    os.environ['REQUESTS_RETRY_DELAY'] = '10'
    os.environ['REQUESTS_MAX_RETRIES'] = '5'
    
    # 设置urllib3的配置
    os.environ['URLLIB3_TIMEOUT'] = '600'
    os.environ['URLLIB3_RETRY_DELAY'] = '10'
    os.environ['URLLIB3_MAX_RETRIES'] = '5'
    
    # 禁用SSL验证警告（如果需要）
    os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
    
    if verbose:
        logging.info(f"已设置Hugging Face缓存目录: {custom_cache_dir}")
        if enable_mirror:
            logging.info(f"已设置HF-Mirror镜像站: https://hf-mirror.com")
        logging.info(f"已设置网络超时: 600秒, 重试次数: 5次, 重试延迟: 10秒")
    
    return custom_cache_dir

def get_cache_dir():
    """
    获取缓存目录路径
    
    Returns:
        str: 缓存目录路径
    """
    from config_loader import load_embedding_config
    embedding_config = load_embedding_config()
    # 优先级：
    # 1. 配置文件中的 cache_dir
    # 2. 项目根目录下的 .hf_cache

    if embedding_config.cache_dir is not None:
        return embedding_config.cache_dir
    else:
        # 获取项目根目录
        core_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(core_dir)
        return os.path.join(project_root, ".hf_cache")

def print_cache_info():
    """打印当前缓存配置信息"""
    logging.info("=== Hugging Face缓存配置信息 ===")
    logging.info(f"HF_HOME: {os.environ.get('HF_HOME', '(未设置)')}")
    logging.info(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '(未设置)')}")
    logging.info(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', '(未设置)')}")
    logging.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(未设置)')}")
    
    cache_dir = get_cache_dir()
    if os.path.exists(cache_dir):
        logging.info(f"✓ 缓存目录存在: {cache_dir}")
        try:
            files = os.listdir(cache_dir)
            logging.info(f"  缓存目录中的文件数量: {len(files)}")
            if files:
                logging.info(f"  文件列表（前10个）: {files[:10]}")
        except Exception as e:
            logging.warning(f"  无法列出缓存目录内容: {e}")
    else:
        logging.info(f"✗ 缓存目录不存在: {cache_dir}")

if __name__ == "__main__":
    print_cache_info()