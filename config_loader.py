#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器模块

支持从 YAML 配置文件加载配置，提供统一的配置管理。

使用示例:
    # 方式1: 从指定文件加载
    config = load_config('config.yaml')
    ob_config = config.get_oceanbase_hybrid_config()
    
    # 方式2: 直接获取配置对象
    from config_loader import load_oceanbase_hybrid_config
    ob_config = load_oceanbase_hybrid_config('config.yaml')
    
    # 方式3: 使用默认配置
    from config import OceanBaseHybridConfig
    ob_config = OceanBaseHybridConfig()
"""

import os
import yaml
from typing import Optional, Dict, Any
import logging

from config import (
    OceanBaseHybridConfig,
    OceanBaseInsertConfig,
    EmbeddingConfig,
    FusionArgs
)


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_file: 配置文件路径，None 则自动查找
        """
        self.config_file = self._find_config_file(config_file)
        self.config_data = self._load_yaml() if self.config_file else {}

    def _find_config_file(self, config_file: Optional[str]) -> Optional[str]:
        """
        查找配置文件
        
        优先级：
        1. 指定的配置文件
        2. 当前目录的 config.yaml
        3. 项目根目录的 config.yaml
        4. None（使用默认配置）
        """
        if config_file:
            if os.path.exists(config_file):
                return config_file
            else:
                raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        # 查找默认配置文件
        search_paths = [
            'config.yaml',  # 当前目录
            os.path.join(os.path.dirname(__file__), 'config.yaml'),  # 项目根目录
            os.path.expanduser('~/.mldr/config.yaml'),  # 用户目录
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path

        return None

    def _load_yaml(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

    def get_oceanbase_hybrid_config(self) -> OceanBaseHybridConfig:
        """获取 OceanBase 混合搜索配置"""
        if not self.config_data or 'oceanbase' not in self.config_data:
            return OceanBaseHybridConfig()

        ob_data = self.config_data['oceanbase']
        return OceanBaseHybridConfig(
            host=ob_data.get('host', '127.0.0.1'),
            port=ob_data.get('port', 2881),
            user=ob_data.get('user', 'root'),
            password=ob_data.get('password', ''),
            database=ob_data.get('database', 'test'),
            charset=ob_data.get('charset', 'utf8mb4'),
            autocommit=ob_data.get('autocommit', True)
        )

    def get_oceanbase_insert_config(self) -> OceanBaseInsertConfig:
        """获取 OceanBase 数据插入配置"""
        if not self.config_data or 'oceanbase' not in self.config_data:
            return OceanBaseInsertConfig()

        ob_data = self.config_data['oceanbase']
        insert_data = ob_data.get('insert', {})

        return OceanBaseInsertConfig(
            # 连接配置
            host=ob_data.get('host', '127.0.0.1'),
            port=ob_data.get('port', 2881),
            user=ob_data.get('user', 'root'),
            password=ob_data.get('password', ''),
            database=ob_data.get('database', 'test'),
            charset=ob_data.get('charset', 'utf8mb4'),
            connect_timeout=ob_data.get('connect_timeout', 60),
            read_timeout=insert_data.get('read_timeout', 600),
            write_timeout=insert_data.get('write_timeout', 600),
            autocommit=insert_data.get('autocommit', False),
            # 插入配置
            batch_size=insert_data.get('batch_size', 1000),
            enable_streaming=insert_data.get('enable_streaming', True),
            # 向量配置
            vector_batch_size=insert_data.get('vector_batch_size', 32),
            use_fp16=insert_data.get('use_fp16', True)
        )

    def get_fusion_args(self) -> FusionArgs:
        """获取融合查询配置"""
        return FusionArgs()

    def get_embedding_config(self) -> EmbeddingConfig:
        """获取嵌入模型配置"""
        if not self.config_data or 'embedding' not in self.config_data:
            return EmbeddingConfig()

        emb_data = self.config_data['embedding']

        # 创建默认配置对象以获取默认值
        default_config = EmbeddingConfig()

        return EmbeddingConfig(
            download_vectors=emb_data.get('download_vectors', default_config.download_vectors),
            vector_download_url=emb_data.get('vector_download_url', default_config.vector_download_url)
        )


# 全局配置加载器单例
_global_loader: Optional[ConfigLoader] = None


def load_config(config_file: Optional[str] = None) -> ConfigLoader:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径，None 则自动查找
    
    Returns:
        ConfigLoader: 配置加载器实例
    
    Example:
        >>> loader = load_config('config.yaml')
        >>> ob_config = loader.get_oceanbase_config()
    """
    return ConfigLoader(config_file)


def set_global_config(config_file: Optional[str] = None):
    """
    设置全局配置文件
    
    Args:
        config_file: 配置文件路径
    
    Example:
        >>> set_global_config('config.yaml')
        >>> # 之后所有 load_xxx_config() 都会使用这个配置
    """
    global _global_loader
    _global_loader = ConfigLoader(config_file)


def get_global_loader() -> ConfigLoader:
    """获取全局配置加载器"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader


# 便捷函数：直接加载各类配置
def load_oceanbase_hybrid_config(config_file: Optional[str] = None) -> OceanBaseHybridConfig:
    """加载 OceanBase 混合搜索配置"""
    if config_file:
        return ConfigLoader(config_file).get_oceanbase_hybrid_config()
    return get_global_loader().get_oceanbase_hybrid_config()


def load_oceanbase_insert_config(config_file: Optional[str] = None) -> OceanBaseInsertConfig:
    """加载 OceanBase 数据插入配置"""
    if config_file:
        return ConfigLoader(config_file).get_oceanbase_insert_config()
    return get_global_loader().get_oceanbase_insert_config()


def load_embedding_config(config_file: Optional[str] = None) -> EmbeddingConfig:
    """加载嵌入模型配置"""
    if config_file:
        return ConfigLoader(config_file).get_embedding_config()
    return get_global_loader().get_embedding_config()


# 导出的公共 API
__all__ = [
    'ConfigLoader',
    'load_config',
    'set_global_config',
    'load_oceanbase_hybrid_config',
    'load_oceanbase_insert_config',
    'load_embedding_config',
]


if __name__ == "__main__":
    pass
