#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 测试框架核心模块

本模块提供以下核心功能：
1. Hugging Face 缓存配置管理 (hf_cache_config)
2. MLDR 数据集加载 (load_mldr_local)
3. 通用工具函数 (mldr_common_tools)
4. 向量嵌入生成和管理 (embeddings)
"""

# ============================================
# 1. HF 缓存配置相关
# ============================================
from .hf_cache_config import (
    setup_hf_cache,
    get_cache_dir,
    print_cache_info
)

setup_hf_cache(enable_mirror=True, verbose=False)

# ============================================
# 2. MLDR 数据加载相关
# ============================================
from .load_mldr_local import (
    load_corpus_local,
    get_queries_and_qids_local,
    download_all_languages,
    available_languages
)

# ============================================
# 3. 通用工具 - 数据类
# ============================================
from .mldr_common_tools import (
    EvalArgs,
    QueryArgs,
    FakeJScoredDoc
)

# ============================================
# 4. 通用工具 - 数据加载函数
# ============================================
from .mldr_common_tools import (
    load_corpus,
    get_queries_and_qids,
    check_languages,
    check_query_types,
    get_default_query_result_dir
)

# ============================================
# 5. 通用工具 - 结果保存函数
# ============================================
from .mldr_common_tools import (
    save_result,
    save_result_with_scores
)

# ============================================
# 6. 通用工具 - 向量读取函数
# ============================================
from .mldr_common_tools import (
    fvecs_read,
    fvecs_read_yield,
    ivecs_read,
    read_mldr_sparse_embedding,
    read_mldr_sparse_embedding_list_dict,
    read_mldr_sparse_embedding_yield,
    read_colbert_data_yield,
    save_colbert_list
)

# ============================================
# 7. 通用工具 - 查询相关
# ============================================
from .mldr_common_tools import (
    query_yields,
    apply_funcs,
    bm25_query_yield,
    dense_query_yield,
    sparse_query_yield,
    colbert_query_yield,
    get_colbert_model
)

# ============================================
# 8. 通用工具 - 辅助函数
# ============================================
from .mldr_common_tools import (
    get_all_part_begin_ends,
    get_bit_array
)

# ============================================
# 9. 向量嵌入生成和管理
# ============================================
from .embeddings import (
    ModelArgs,
    VectorFileIterator,
    get_bge_m3_model,
    prepare_dense_embedding,
    # prepare_sparse_embedding,
    # prepare_colbert_embedding,
    load_dense_vectors_from_file,
    save_dense_vectors_to_file,
    generate_and_save_dense_vectors,
    generate_dense_vectors_from_stream,
    download_dense_vectors,
    generate_vector_filename
)


# ============================================
# 定义包的公共 API
# ============================================
__all__ = [
    # HF 缓存配置
    'setup_hf_cache',
    'get_cache_dir',
    'print_cache_info',
    
    # MLDR 数据加载
    'load_corpus_local',
    'get_queries_and_qids_local',
    'download_all_languages',
    'available_languages',
    
    # 数据类
    'EvalArgs',
    'QueryArgs',
    'FakeJScoredDoc',
    
    # 数据加载
    'load_corpus',
    'get_queries_and_qids',
    'check_languages',
    'check_query_types',
    'get_default_query_result_dir',
    
    # 结果保存
    'save_result',
    'save_result_with_scores',
    
    # 向量读取
    'fvecs_read',
    'fvecs_read_yield',
    'ivecs_read',
    'read_mldr_sparse_embedding',
    'read_mldr_sparse_embedding_list_dict',
    'read_mldr_sparse_embedding_yield',
    'read_colbert_data_yield',
    'save_colbert_list',
    
    # 查询相关
    'query_yields',
    'apply_funcs',
    'bm25_query_yield',
    'dense_query_yield',
    'sparse_query_yield',
    'colbert_query_yield',
    'get_colbert_model',
    
    # 辅助函数
    'get_all_part_begin_ends',
    'get_bit_array',
    
    # 向量嵌入
    'ModelArgs',
    'VectorFileIterator',
    'get_bge_m3_model',
    'prepare_dense_embedding',
    # 'prepare_sparse_embedding',
    # 'prepare_colbert_embedding',
    'load_dense_vectors_from_file',
    'save_dense_vectors_to_file',
    'generate_and_save_dense_vectors',
    'generate_dense_vectors_from_stream',
    'download_dense_vectors',
    'generate_vector_filename'
]

import logging
# ============================================
# 便捷函数：一键初始化
# ============================================
def initialize_mldr_env(enable_hf_cache: bool = True, verbose: bool = True):
    """
    一键初始化 MLDR 测试环境
    
    Args:
        enable_hf_cache: 是否启用 HF 缓存配置
        verbose: 是否打印详细信息
    
    Returns:
        dict: 初始化信息
    """
    info = {
        'hf_cache_enabled': False,
        'cache_dir': None,
        'available_languages': available_languages
    }
    
    if enable_hf_cache:
        cache_dir = setup_hf_cache()
        info['hf_cache_enabled'] = True
        info['cache_dir'] = cache_dir
        
        if verbose:
            logging.info("✓ MLDR 测试环境初始化完成")
            logging.info(f"  - HF 缓存目录: {cache_dir}")
            logging.info(f"  - 支持语言数: {len(available_languages)}")
    else:
        if verbose:
            logging.info("✓ MLDR 测试环境初始化完成（未启用 HF 缓存）")
    
    return info


if __name__ == "__main__":
    # 测试包导入
    logging.info(f"可用的公共 API 数量: {len(__all__)}")
    logging.info("\n测试初始化环境...")
    info = initialize_mldr_env(enable_hf_cache=False, verbose=True)
    logging.info(f"\n初始化信息: {info}")