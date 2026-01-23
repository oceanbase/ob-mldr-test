#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 测试框架数据库操作模块

本模块提供以下功能：
1. OceanBase 数据插入和索引创建 (insert_oceanbase)
"""

# ============================================
# OceanBase 数据插入
# ============================================
from .insert_oceanbase import (
    test_simple_insert as insert_data_oceanbase,
    main as oceanbase_insert_main
)

# ============================================
# 定义包的公共 API
# ============================================
__all__ = [
    'insert_data_oceanbase',
    'oceanbase_insert_main',
]

if __name__ == "__main__":
    # 测试包导入
    import logging
    logging.info(f"可用的公共 API 数量: {len(__all__)}")