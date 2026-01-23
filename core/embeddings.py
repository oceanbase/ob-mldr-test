#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量嵌入生成模块

提供统一的向量生成功能，支持：
- Dense vectors (BGE-M3)
- Sparse vectors (BGE-M3)
- ColBERT vectors
"""

import struct
import os
import urllib.request
from .hf_cache_config import get_cache_dir
from huggingface_hub import snapshot_download

import subprocess
import numpy as np
from dataclasses import dataclass, field
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import logging
from typing import Optional, List, Iterator

from . import get_colbert_model, save_colbert_list

class VectorFileIterator:
    """
    .fvecs 文件的高效迭代器，支持按批次读取
    单次顺序遍历，O(N) 时间，低内存
    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"向量文件不存在: {filepath}")
        self.filepath = filepath
        self._file_handle = None

    def __enter__(self):
        self._file_handle = open(self.filepath, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()

    def __iter__(self):
        return self

    def __next__(self) -> list:
        """返回单个向量（list of float）"""
        dim_bytes = self._file_handle.read(4)
        if not dim_bytes:
            raise StopIteration
        
        dim = struct.unpack('i', dim_bytes)[0]
        vector_bytes = self._file_handle.read(dim * 4)
        if len(vector_bytes) < dim * 4:
            raise RuntimeError("向量文件损坏或不完整")
        
        vector = np.frombuffer(vector_bytes, dtype=np.float32).tolist()
        return vector

    def read_batch(self, batch_size: int) -> List[list]:
        """读取一个批次（最多 batch_size 个向量）"""
        batch = []
        for _ in range(batch_size):
            try:
                vec = self.__next__()
                batch.append(vec)
            except StopIteration:
                break
        return batch


@dataclass
class ModelArgs:
    """模型参数配置（嵌入模型）"""
    # 主模型配置
    model_name_or_path: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'The name or path of the main embedding model.'}
    )
    colbert_model: str = field(
        default="jina-colbert", 
        metadata={'help': 'The name of the colbert model to use.'}
    )
    
    # 精度配置
    fp16: bool = field(
        default=True, 
        metadata={'help': 'Use fp16 in inference?'}
    )
    use_fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 (alias for fp16 for consistency)'}
    )
    
    # 嵌入配置
    pooling_method: str = field(
        default="cls",
        metadata={'help': 'Pooling method: cls, mean, max'}
    )
    normalize_embeddings: bool = field(
        default=False,
        metadata={'help': 'Whether to normalize embeddings'}
    )
    max_length: int = field(
        default=8192,
        metadata={'help': 'Maximum sequence length'}
    )
    
    # 批处理配置
    batch_size: int = field(
        default=32,
        metadata={'help': 'Batch size for encoding'}
    )
    
    def __post_init__(self):
        """确保 fp16 和 use_fp16 保持一致"""
        # 如果两个都设置了，use_fp16 优先
        if hasattr(self, 'use_fp16'):
            self.fp16 = self.use_fp16


def get_bge_m3_model(model_args: ModelArgs):
    """
    获取 BGE-M3 模型
    
    Args:
        model_args: 模型参数配置
    
    Returns:
        BGEM3FlagModel 实例
    """
    local_model_path = snapshot_download(
        repo_id="BAAI/bge-m3",
        cache_dir=get_cache_dir(),
        ignore_patterns=["*.DS_Store", "imgs/*", "*.md", "LICENSE"]
    )
    return BGEM3FlagModel(local_model_path, use_fp16=model_args.fp16)


def prepare_dense_embedding(
    embedding_file: str, 
    model_args: ModelArgs, 
    queries: list[str], 
    qids: list[int]
):
    """
    生成密集向量嵌入并保存到文件
    
    Args:
        embedding_file: 保存路径
        model_args: 模型参数
        queries: 查询文本列表
        qids: 查询ID列表
    
    Raises:
        AssertionError: 如果向量维度不匹配
    """
    model = get_bge_m3_model(model_args)
    query_embedding = model.encode(
        queries, 
        return_dense=True, 
        return_sparse=False, 
        return_colbert_vecs=False
    )
    query_embedding = query_embedding['dense_vecs'].astype(np.float32)
    
    # 验证维度
    assert len(query_embedding.shape) == 2, "Expected 2D array"
    assert query_embedding.shape[0] == len(qids), "Embedding count mismatch"
    assert query_embedding.shape[1] == 1024, "Expected 1024 dimensions"
    
    # 保存为 fvecs 格式
    with open(embedding_file, 'wb') as f:
        for single_embedding in query_embedding:
            assert len(single_embedding) == 1024
            f.write(struct.pack('i', 1024))
            single_embedding.tofile(f)
    
    logging.debug(f"✓ 密集向量已保存: {embedding_file} ({len(qids)} 条)")


# def prepare_sparse_embedding(
#     embedding_file: str, 
#     model_args: ModelArgs, 
#     queries: list[str], 
#     qids: list[int]
# ):
#     """
#     生成稀疏向量嵌入并保存到文件
    
#     Args:
#         embedding_file: 保存路径
#         model_args: 模型参数
#         queries: 查询文本列表
#         qids: 查询ID列表
    
#     Raises:
#         AssertionError: 如果向量数量不匹配
#     """
#     model = get_bge_m3_model(model_args)
#     query_embedding = model.encode(
#         queries, 
#         return_dense=False, 
#         return_sparse=True, 
#         return_colbert_vecs=False
#     )
#     query_embedding = query_embedding['lexical_weights']
    
#     assert len(query_embedding) == len(qids), "Embedding count mismatch"
    
#     # 保存为自定义二进制格式
#     with open(embedding_file, 'wb') as f:
#         f.write(struct.pack('i', len(query_embedding)))
#         for one_dict in query_embedding:
#             # 转换为排序的 (index, value) 列表
#             tmp_list = []
#             for p, v in one_dict.items():
#                 tmp_list.append((int(p), float(v)))
#             tmp_list.sort()
            
#             # 写入
#             f.write(struct.pack('i', len(tmp_list)))
#             for p, v in tmp_list:
#                 f.write(struct.pack('if', p, v))
    
#     logging.debug(f"✓ 稀疏向量已保存: {embedding_file} ({len(qids)} 条)")


# def prepare_colbert_embedding(
#     embedding_file: str, 
#     model_args: ModelArgs, 
#     queries: list[str], 
#     qids: list[int]
# ):
#     """
#     生成 ColBERT 向量嵌入并保存到文件
    
#     Args:
#         embedding_file: 保存路径
#         model_args: 模型参数
#         queries: 查询文本列表
#         qids: 查询ID列表
    
#     Raises:
#         AssertionError: 如果向量数量不匹配
#     """
#     model = get_colbert_model(model_args)
#     query_embedding = model.encode_query(queries)
#     query_embedding = query_embedding['colbert_vecs']
    
#     assert len(query_embedding) == len(qids), "Embedding count mismatch"
    
#     # 使用 core 模块的保存函数
#     save_colbert_list(query_embedding, embedding_file)
    
#     logging.debug(f"✓ ColBERT向量已保存: {embedding_file} ({len(qids)} 条)")


# ============================================
# 向量文件管理和验证
# ============================================

def generate_vector_filename(lang: str, count: int, model_name: str = 'bge-m3') -> str:
    """
    生成安全的向量文件名

    Args:
        lang: 语言代码
        count: 文本数量
        model_name: 模型名称

    Returns:
        str: 向量文件名（不含路径）
    """
    return f"corpus_dense_{lang}_{count}_{model_name}.fvecs"


def load_dense_vectors_from_file(vector_file: str, expected_count: int = None) -> list:
    """
    从 fvecs 文件加载密集向量
    
    Args:
        vector_file: 向量文件路径
        expected_count: 期望的向量数量（None 表示加载所有）
    
    Returns:
        list: 向量列表
    """
    vectors = []
    with open(vector_file, 'rb') as f:
        while expected_count is None or len(vectors) < expected_count:
            # 读取维度
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            
            dim = struct.unpack('i', dim_bytes)[0]
            # 读取向量数据
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) < dim * 4:
                break
            
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector.tolist())
    
    return vectors

# ============================================
# 向量文件读写工具
# ============================================


def save_dense_vectors_to_file(vectors: list, vector_file: str):
    """
    保存密集向量到 fvecs 文件
    
    Args:
        vectors: 向量列表
        vector_file: 保存路径
    """
    os.makedirs(os.path.dirname(vector_file) or '.', exist_ok=True)
    with open(vector_file, 'wb') as f:
        for vector in vectors:
            vector_array = np.array(vector, dtype=np.float32)
            dim = len(vector_array)
            f.write(struct.pack('i', dim))
            vector_array.tofile(f)


# ============================================
# 批量向量生成工具
# ============================================
def generate_and_save_dense_vectors(
    texts: List[str],
    output_file: str,
    model_args: ModelArgs,
    batch_size: int = 32,
    show_progress: bool = True
) -> str:
    """
    批量生成密集向量并直接保存到 .fvecs 文件（流式写入，低内存）
    
    Args:
        texts: 文本列表（可全量加载，但向量不全量保留）
        output_file: 输出 .fvecs 文件路径
        model_args: 模型参数
        batch_size: 批次大小
        show_progress: 是否显示进度条
    
    Returns:
        str: 输出文件路径
    
    Note:
        - 文本需全量加载（因 BGE-M3 需要 list 输入）
        - 向量分批生成并立即写入磁盘，内存仅保留当前批次
    """
    if os.path.exists(output_file):
        logging.debug(f"本地向量文件已存在: {output_file}")
        file_size = os.path.getsize(output_file)
        logging.debug(f"文件大小: {file_size / (1024*1024):.2f} MB")
        return output_file

    logging.debug(f"   加载模型: BAAI/bge-m3 (fp16={model_args.fp16})")
    model = get_bge_m3_model(model_args)
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)
    
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="   生成并保存向量",
            total=total_batches,
            unit="batch",
            ncols=80
        )
    
    with open(output_file, 'wb') as f:
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            batch_vectors = batch_embeddings['dense_vecs'].astype(np.float32)
            
            # 立即写入文件
            for vec in batch_vectors:
                dim = len(vec)
                f.write(struct.pack('i', dim))
                vec.tofile(f)
    
    logging.debug(f"✓ 向量已保存到: {output_file} ({len(texts)} 条)")
    return output_file

# ============================================
# 向量文件下载工具
# ============================================
def download_dense_vectors(
    lang: str,
    vector_dir: str = "vectors",
    base_url: str = None,
    vector_range: str = "0-200000"
) -> str:
    """
    下载预生成的密集向量文件
    
    Args:
        lang: 语言代码
        vector_dir: 本地保存目录
        base_url: 下载服务器URL（None则从配置文件读取）
        vector_range: 向量范围（用于文件名）
    
    Returns:
        str: 下载后的文件路径
    
    Raises:
        Exception: 如果下载失败
    """
    # 如果未指定base_url，从配置文件读取
    if base_url is None:
        from config_loader import load_embedding_config
        embedding_config = load_embedding_config()
        base_url = embedding_config.vector_download_url
        logging.debug(f"从配置文件读取向量下载URL: {base_url}")
    
    # 验证URL不为空
    if not base_url or base_url.strip() == "":
        raise ValueError(
            "向量下载URL未配置！\n"
            "请在 config.yaml 中配置 embedding.vector_download_url\n"
            "例如：\n"
            "embedding:\n"
            "  download_vectors: true\n"
            "  vector_download_url: \"http://your-server.com/path/to/vectors\""
        )
    
    # 构建本地保存路径
    local_dense_dir = os.path.join(vector_dir, lang)
    # 使用标准的向量文件名格式
    count = int(vector_range.split('-')[1]) if '-' in vector_range else 200000
    dense_file_path = os.path.join(local_dense_dir, generate_vector_filename(lang, count, 'bge-m3'))

    # 如果文件已存在，直接返回
    if os.path.exists(dense_file_path):
        logging.debug(f"本地向量文件已存在: {dense_file_path}")
        file_size = os.path.getsize(dense_file_path)
        logging.debug(f"文件大小: {file_size / (1024*1024):.2f} MB")
        return dense_file_path
    
    # 创建目录
    os.makedirs(local_dense_dir, exist_ok=True)
    logging.debug(f"创建目录: {local_dense_dir}")
    
    # 构建下载URL
    download_url = f"{base_url}/{lang}/dense-{vector_range}.fvecs"
    logging.debug(f"开始下载向量文件...")
    logging.debug(f"  URL: {download_url}")
    logging.debug(f"  保存: {dense_file_path}")
    
    # 执行下载
    try:
        logging.debug(f"开始下载向量文件...")
        logging.debug(f"  URL: {download_url}")
        logging.debug(f"  保存: {dense_file_path}")

        # 使用urllib下载并显示进度条
        def download_with_progress(url, filepath):
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                block_size = 8192

                with open(filepath, 'wb') as f:
                    with tqdm(
                        desc="   下载向量文件",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        ncols=80
                    ) as pbar:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            f.write(buffer)
                            pbar.update(len(buffer))

        download_with_progress(download_url, dense_file_path)

        logging.debug("✓ 下载完成！")

        # 验证文件
        if os.path.exists(dense_file_path):
            file_size = os.path.getsize(dense_file_path)
            logging.debug(f"✓ 文件大小: {file_size / (1024*1024):.2f} MB")
            return dense_file_path
        else:
            raise FileNotFoundError("下载后文件不存在")

    except Exception as e:
        logging.error(f"✗ 下载失败: {e}")
        # 清理不完整的文件
        if os.path.exists(dense_file_path):
            os.remove(dense_file_path)
        raise

def generate_dense_vectors_from_stream(
    text_stream: Iterator,
    output_file: str,
    model_args: ModelArgs,
    batch_size: int = 32,
    total: Optional[int] = None,
    show_progress: bool = True
) -> str:
    """
    从文本流生成向量并保存到文件（真正的流式处理）
    
    Args:
        text_stream: 文本迭代器（如 load_corpus(lang, streaming=True)）
        output_file: 输出 .fvecs 文件路径
        model_args: 模型参数
        batch_size: 内部批处理大小
        total: 总条数（用于进度条）
        show_progress: 是否显示进度条
    
    Returns:
        输出文件路径
    """
    def _encode_batch(model, texts: List[str]) -> np.ndarray:
        """内部批量编码"""
        embeddings = model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return embeddings['dense_vecs'].astype(np.float32)

    def _write_vectors_to_file(f, vectors: np.ndarray):
        """写入 .fvecs 格式"""
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack('i', dim))
            vec.tofile(f)

    if os.path.exists(output_file):
        logging.debug(f"本地向量文件已存在: {output_file}")
        file_size = os.path.getsize(output_file)
        logging.debug(f"文件大小: {file_size / (1024*1024):.2f} MB")
        return output_file

    model = get_bge_m3_model(model_args)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    buffer = []
    total_num = 0
    
    with open(output_file, 'wb') as f_out:
        pbar = tqdm(desc="生成向量", total=total, unit="条") if show_progress else None
        
        processed = 0
        for text in text_stream:
            if total is not None and processed >= total:
                break
            buffer.append(text["text"])
            processed += 1

            if len(buffer) >= batch_size:
                vectors = _encode_batch(model, buffer)
                _write_vectors_to_file(f_out, vectors)
                if pbar:
                    pbar.update(len(buffer))
                buffer.clear()
        
        # 处理剩余
        if buffer:
            vectors = _encode_batch(model, buffer)
            _write_vectors_to_file(f_out, vectors)
            if pbar:
                pbar.update(len(buffer))
        
        if pbar:
            pbar.close()
    
    return output_file

# 导出的公共API
__all__ = [
    'ModelArgs',
    'VectorFileIterator',
    'get_bge_m3_model',
    'prepare_dense_embedding',
    # 'prepare_sparse_embedding',
    # 'prepare_colbert_embedding',
    # 向量文件读写
    'load_dense_vectors_from_file',
    'save_dense_vectors_to_file',
    # 批量向量生成
    'generate_and_save_dense_vectors',
    'generate_dense_vectors_from_stream',
    # 向量文件下载
    'download_dense_vectors',
    'generate_vector_filename'
]
