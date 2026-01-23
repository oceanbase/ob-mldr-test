#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation 模块 - 搜索结果评估

提供 MLDR 数据集搜索结果的评估功能。
"""

from .evaluate_recall import (
    evaluate_search_results,
    check_qrels_files,
    main as evaluate_main
)

# from .qps_benchmark import (
#     QPSBenchmarkVectorDBBenchAligned,
#     QPSConfig,
#     VectorDBBenchAlignedSearchRunner,
#     main as qps_benchmark_main
# )

__all__ = [
    # 评估功能
    'evaluate_search_results',
    'check_qrels_files',
    'evaluate_main',

    # QPS 基准测试
    # 'QPSBenchmarkVectorDBBenchAligned',
    # 'QPSConfig',
    # 'VectorDBBenchAlignedSearchRunner',
    # 'qps_benchmark_main',
]
