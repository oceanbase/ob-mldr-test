#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成密集向量嵌入工具

使用示例:
python generate_dense_embedding.py \
    --begin_pos 0 \
    --end_pos 200000 \
    --languages ar de en es fr hi it ja ko pt ru th zh \
    --embedding_save_dir ./corpus-embedding \
    --max_passage_length 8192 \
    --batch_size 1 \
    --fp16
"""
import os
import struct
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from FlagEmbedding import FlagModel
import logging

from core import (
    EvalArgs,
    check_languages,
    load_corpus
)
from core.embeddings import (
    ModelArgs,
    save_dense_vectors_to_file
)


def get_model(model_args: ModelArgs):
    """
    获取密集向量模型（支持自定义 encoder）
    
    Args:
        model_args: 模型参数配置
    
    Returns:
        FlagModel: 初始化的模型实例
    
    Note:
        这里保留独立实现是因为 tools 支持自定义 encoder，
        而 core.embeddings 固定使用 BGE-M3
    """
    model = FlagModel(
        model_args.encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model


@dataclass
class ToolModelArgs:
    """工具专用的模型参数配置（支持自定义encoder）"""
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Available methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


def generate_dense(
    model: FlagModel,
    corpus,
    max_passage_length: int,
    batch_size: int,
    begin_pos: int,
    end_pos: int
):
    """
    生成密集向量嵌入
    
    Args:
        model: FlagModel 实例
        corpus: 语料库数据集
        max_passage_length: 最大文本长度
        batch_size: 批次大小
        begin_pos: 开始位置
        end_pos: 结束位置
    
    Returns:
        list: 密集向量列表
    """
    logging.debug(f"开始生成向量: 位置 {begin_pos} 到 {end_pos}")
    logging.debug(f"批次大小: {batch_size}, 最大长度: {max_passage_length}")
    
    # 计算总批次数
    total_samples = end_pos - begin_pos
    total_batches = (total_samples + batch_size - 1) // batch_size
    logging.debug(f"总样本数: {total_samples}, 总批次数: {total_batches}")

    # 使用tqdm显示编码进度
    texts = corpus["text"][begin_pos:end_pos]
    dense_embeddings = model.encode_corpus(
        texts,
        batch_size=batch_size,
        max_length=max_passage_length
    )
    dense_embeddings = dense_embeddings.astype(np.float32)
    logging.debug(f"✓ 向量生成完成，形状: {dense_embeddings.shape}")
    
    # 转换为列表格式，与 core.embeddings 统一
    return dense_embeddings.tolist()


def save_result(dense_embeddings, dense_save_file: str):
    """
    保存密集向量到 fvecs 文件（复用 core.embeddings 的实现）
    
    Args:
        dense_embeddings: 向量列表
        dense_save_file: 保存文件路径
    """
    # 使用 core.embeddings 的统一保存函数
    save_dense_vectors_to_file(dense_embeddings, dense_save_file)
    
    file_size = os.path.getsize(dense_save_file)
    file_size_mb = file_size / (1024 * 1024)
    logging.debug(f"✓ 向量保存完成: {dense_save_file}")
    logging.debug(f"  文件大小: {file_size_mb:.2f} MB")


def main():
    """主函数"""
    parser = HfArgumentParser([ToolModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ToolModelArgs
    eval_args: EvalArgs

    languages = check_languages(eval_args.languages)

    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]

    model = get_model(model_args=model_args)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)

    logging.debug("=" * 60)
    logging.debug("开始生成密集向量")
    logging.debug(f"模型: {model_args.encoder}")
    logging.debug(f"语言: {languages}")
    logging.debug("=" * 60)

    for lang in languages:
        logging.debug(f"\n{'*' * 60}")
        logging.debug(f"处理语言: {lang}")
        logging.debug('*' * 60)
        
        embedding_save_dir = os.path.join(
            eval_args.embedding_save_dir,
            os.path.basename(encoder),
            lang
        )
        os.makedirs(embedding_save_dir, exist_ok=True)
        
        dense_save_file = os.path.join(
            embedding_save_dir,
            f'dense-{eval_args.begin_pos}-{eval_args.end_pos}.fvecs'
        )
        
        if os.path.exists(dense_save_file) and not eval_args.overwrite:
            logging.debug(f'✓ 向量文件已存在: {dense_save_file}')
            logging.debug('  跳过生成（使用 --overwrite 强制重新生成）')
            continue

        logging.debug(f"加载语料库: {lang}")
        corpus = load_corpus(lang)

        logging.debug(f"生成密集向量...")
        dense_embeddings = generate_dense(
            model=model,
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size,
            begin_pos=eval_args.begin_pos,
            end_pos=eval_args.end_pos
        )
        
        save_result(dense_embeddings, dense_save_file)

    logging.debug(f"\n{'=' * 60}")
    logging.debug("✓ 所有语言的密集向量生成完成！")
    logging.debug(f"模型: {model_args.encoder}")
    logging.debug("=" * 60)


if __name__ == "__main__":
    main()
