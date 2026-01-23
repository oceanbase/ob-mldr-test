#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索客户端配置模块

提供统一的配置管理，支持：
- OceanBase 混合搜索配置
- OceanBase 数据插入配置
- 嵌入模型配置
- 融合查询配置
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OceanBaseHybridConfig:
    """OceanBase 混合搜索客户端配置"""
    host: str = '127.0.0.1'
    port: int = 2881
    user: str = 'root'
    password: str = ''
    database: str = 'test'
    charset: str = 'utf8mb4'
    autocommit: bool = True


@dataclass
class OceanBaseInsertConfig:
    """OceanBase 数据插入配置"""
    # 数据库连接配置
    host: str = '127.0.0.1'
    port: int = 2881
    user: str = 'root'
    password: str = ''
    database: str = 'test'
    charset: str = 'utf8mb4'
    connect_timeout: int = 60
    read_timeout: int = 600
    write_timeout: int = 600
    autocommit: bool = False

    # 插入配置
    batch_size: int = 1000
    test_count: int = 200000
    enable_streaming: bool = True

    # 向量生成配置（当需要向量时使用）
    vector_batch_size: int = 32  # 向量生成批次大小
    use_fp16: bool = True  # 使用 FP16 精度
    
    def to_db_config(self) -> dict:
        """转换为数据库连接配置字典"""
        return {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'charset': self.charset,
            'connect_timeout': self.connect_timeout,
            'read_timeout': self.read_timeout,
            'write_timeout': self.write_timeout,
            'autocommit': self.autocommit
        }


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    # 主模型配置
    model_name_or_path: str = "BAAI/bge-m3"
    colbert_model: str = "jina-colbert"

    # 精度配置
    use_fp16: bool = True

    # 嵌入配置
    pooling_method: str = "cls"
    normalize_embeddings: bool = False
    max_length: int = 8192

    # 批处理配置
    batch_size: int = 32

    # 缓存配置
    cache_dir: Optional[str] = None

    # 向量下载配置
    download_vectors: bool = False  # 是否从网上下载预生成的向量文件，不开启则本地生成
    vector_download_url: str = ""  # 向量文件下载URL（必填：当 download_vectors=true 时必须配置）


@dataclass
class FusionArgs:
    """融合查询参数配置"""
    fusion_method: str = field(
        default="rrf",
        metadata={'help': 'Fusion method: rrf, weighted, or normalized'}
    )
    weights: Optional[str] = field(
        default=None,
        metadata={'help': 'Weights for weighted fusion (comma-separated, e.g., "0.6,0.4")'}
    )


# 导出的公共API
__all__ = [
    'OceanBaseHybridConfig',
    'OceanBaseInsertConfig',
    'EmbeddingConfig',
    'FusionArgs',
]
