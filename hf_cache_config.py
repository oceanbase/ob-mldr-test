#!/usr/bin/env python3
"""
Hugging Face缓存配置模块
统一设置所有Hugging Face相关的缓存路径到自定义目录
"""

import os
import sys

def setup_hf_cache():
    """设置Hugging Face缓存路径到自定义目录"""
    
    # 自定义缓存目录：使用当前目录下的 .hf_cache 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    custom_cache_dir = os.path.join(current_dir, ".hf_cache")
    
    # 创建缓存目录
    os.makedirs(custom_cache_dir, exist_ok=True)
    
    # 设置所有Hugging Face相关的环境变量
    os.environ['HF_HOME'] = custom_cache_dir
    os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = custom_cache_dir
    os.environ['HF_HUB_CACHE'] = custom_cache_dir
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    # 设置HF-Mirror镜像站 - 更全面的配置
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_BASE_URL'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    # 设置模型和数据集镜像
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
    
    print(f"已设置Hugging Face缓存目录: {custom_cache_dir}")
    print(f"已设置HF-Mirror镜像站: https://hf-mirror.com")
    print(f"已设置网络超时: 600秒, 重试次数: 5次, 重试延迟: 10秒")
    
    return custom_cache_dir

def get_cache_dir():
    """获取缓存目录路径"""
    # 如果环境变量已设置，使用环境变量；否则使用当前目录下的 .hf_cache
    if 'HF_HOME' in os.environ:
        return os.environ.get('HF_HOME')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, ".hf_cache")

def print_cache_info():
    """打印当前缓存配置信息"""
    print("=== Hugging Face缓存配置信息 ===")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    
    cache_dir = get_cache_dir()
    if os.path.exists(cache_dir):
        print(f"缓存目录存在: {cache_dir}")
        try:
            files = os.listdir(cache_dir)
            print(f"缓存目录中的文件数量: {len(files)}")
            print(f"文件列表: {files[:10]}...")  # 只显示前10个文件
        except Exception as e:
            print(f"无法列出缓存目录内容: {e}")
    else:
        print(f"缓存目录不存在: {cache_dir}")

# 在模块导入时自动设置缓存
setup_hf_cache()

if __name__ == "__main__":
    print_cache_info() 