#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 测试框架搜索模块

本模块提供以下搜索功能：
1. OceanBase 原生混合搜索（Hybrid Search）- oceanbase_hybrid
2. 支持多种查询类型（BM25/Dense/ColBERT/Hybrid等）

注意：配置类和向量生成工具已移至项目根目录的 config.py 和 embeddings.py
"""


# ============================================
# OceanBase 混合搜索客户端
# ============================================
from .oceanbase_hybrid import (
    OceanBaseClientForHybridSearch,
    main as oceanbase_hybrid_main
)

# ============================================
# 定义包的公共 API
# ============================================
__all__ = [
    'OceanBaseClientForHybridSearch',
    'oceanbase_hybrid_main',
    'QueryType',
]


# ============================================
# 查询类型定义
# ============================================
class QueryType:
    """查询类型常量"""
    # 单查询类型
    BM25 = "bm25"
    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"

    # 混合查询类型
    HYBRID_DENSE_BM25 = "hybrid_dense_bm25"

    # 融合查询类型（RRF）
    FUSION_RRF = "fusion_rrf"
    FUSION_WEIGHTED = "fusion_weighted"

    @classmethod
    def get_all_single_types(cls):
        """获取所有单查询类型"""
        return [cls.BM25, cls.DENSE, cls.SPARSE, cls.COLBERT]

    @classmethod
    def get_all_hybrid_types(cls):
        """获取所有混合查询类型"""
        return [cls.HYBRID_DENSE_BM25]

    @classmethod
    def get_all_fusion_types(cls):
        """获取所有融合查询类型"""
        return [cls.FUSION_RRF, cls.FUSION_WEIGHTED]


if __name__ == "__main__":
    pass