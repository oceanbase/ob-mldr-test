#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 搜索结果评估模块

使用 TREC 评估工具评估搜索结果的质量（Recall, NDCG 等指标）
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pyserini.util import download_evaluation_script

from core import check_languages
import logging


@dataclass
class EvalArgs:
    """评估参数配置"""
    languages: str = field(
        default="en",
        metadata={
            'help': 'Languages to evaluate. Available: ar de en es fr hi it ja ko pt ru th zh',
            "nargs": "+"
        }
    )
    query_result_dir: str = field(
        default='',
        metadata={'help': 'Directory containing query result files'}
    )
    qrels_dir: str = field(
        default='./qrels',
        metadata={'help': 'Directory containing qrels files'}
    )
    metrics: str = field(
        default="ndcg@10",
        metadata={
            'help': 'Metrics to evaluate. Available: ndcg@k, recall@k',
            "nargs": "+"
        }
    )


def map_metric(metric: str):
    """
    映射评估指标到 TREC 格式
    
    Args:
        metric: 指标名称 (e.g., 'ndcg@10', 'recall@10')
    
    Returns:
        Tuple[str, str]: (k值, TREC指标名称)
    """
    metric, k = metric.split('@')
    if metric.lower() == 'ndcg':
        return k, f'ndcg_cut.{k}'
    elif metric.lower() == 'recall':
        return k, f'recall.{k}'
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate(script_path: str, qrels_path: str, query_result_path: str, metrics: list) -> dict:
    """
    使用 TREC 评估工具评估搜索结果
    
    Args:
        script_path: TREC评估工具路径
        qrels_path: Qrels文件路径
        query_result_path: 查询结果文件路径
        metrics: 评估指标列表
    
    Returns:
        dict: 评估结果字典
    """
    cmd_prefix = ['java', '-jar', script_path]
    results = {}
    
    for metric in metrics:
        k, mapped_metric = map_metric(metric)
        args = ['-c', '-M', str(k), '-m', mapped_metric, qrels_path, query_result_path]
        cmd = cmd_prefix + args

        logging.debug(f'Running evaluation command: {" ".join(cmd)}')
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        stdout, stderr = process.communicate()
        
        if stderr:
            logging.warning(f"  Warning: {stderr.decode('utf-8')}")
        
        result_str = stdout.decode("utf-8")
        try:
            results[metric] = float(result_str.split(' ')[-1].split('\t')[-1])
        except Exception as e:
            logging.error(f"  Failed to parse result: {e}")
            logging.info(f"  Raw output: {result_str}")
            results[metric] = result_str
    
    return results


def evaluate_search_results(qrels_file: str, results_file: str, metrics: list = None) -> dict:
    """
    评估搜索结果（简化接口）
    
    Args:
        qrels_file: Qrels文件路径
        results_file: 结果文件路径
        metrics: 评估指标列表 (默认: ['recall_10'])
    
    Returns:
        dict: 评估结果
    """
    if metrics is None:
        metrics = ['recall@10']
    
    # 下载 TREC 评估工具
    script_path = download_evaluation_script('trec_eval')
    
    # 执行评估
    return evaluate(script_path, qrels_file, results_file, metrics)


def check_qrels_files(qrels_dir: str, languages: list, auto_download: bool = True) -> bool:
    """
    检查 qrels 文件是否存在，自动下载缺失的文件

    Args:
        qrels_dir: Qrels 文件目录
        languages: 语言列表
        auto_download: 是否自动下载缺失的文件

    Returns:
        bool: 所有文件都存在返回 True，否则返回 False
    """
    import requests

    missing_files = []
    for lang in languages:
        qrels_path = os.path.join(qrels_dir, f"qrels.mldr-v1.0-{lang}-test.tsv")
        if not os.path.exists(qrels_path):
            missing_files.append((qrels_path, lang))

    if not missing_files:
        return True

    if not auto_download:
        logging.info("✗ Missing qrels files:")
        for file_path, lang in missing_files:
            logging.info(f"  - {file_path}")
        logging.info("\nPlease download qrels files manually:")
        for _, lang in missing_files:
            logging.info(f"   wget https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/qrels/qrels.mldr-v1.0-{lang}-test.tsv")
        return False

    # 自动下载缺失的文件
    logging.debug("🔄 Auto-downloading missing qrels files...")
    os.makedirs(qrels_dir, exist_ok=True)

    success_count = 0
    for qrels_path, lang in missing_files:
        # MLDR 数据集中的原始文件名是 -test.tsv
        remote_filename = f"qrels.mldr-v1.0-{lang}-test.tsv"
        url = f"https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/qrels/{remote_filename}"
        logging.debug(f"  Downloading: {remote_filename} → {os.path.basename(qrels_path)}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(qrels_path, 'wb') as f:
                f.write(response.content)

            logging.debug(f"    ✓ Downloaded and saved as: {os.path.basename(qrels_path)}")
            success_count += 1

        except requests.RequestException as e:
            logging.error(f"    ✗ Failed to download {remote_filename}: {e}")
            logging.info(f"      You can manually download from: {url}")
            logging.info(f"      Then save as: {os.path.basename(qrels_path)}")

        except Exception as e:
            logging.error(f"    ✗ Error saving {os.path.basename(qrels_path)}: {e}")

    if success_count == len(missing_files):
        logging.debug(f"✓ All {success_count} qrels files downloaded successfully")
        return True
    else:
        logging.warning(f"⚠️ Downloaded {success_count}/{len(missing_files)} qrels files")
        logging.info("Some files are still missing. Please check the download URLs above.")
        return False


def main():
    """主函数 - 命令行接口"""
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    # 处理查询结果目录
    if not eval_args.query_result_dir:
        logging.error("✗ Error: --query_result_dir is required")
        return
    
    query_result_dir = eval_args.query_result_dir
    
    languages = check_languages(eval_args.languages)
    metrics = eval_args.metrics if isinstance(eval_args.metrics, list) else [eval_args.metrics]
    
    logging.info("=" * 70)
    logging.info("MLDR Search Results Evaluation")
    logging.info("=" * 70)
    logging.info(f"Languages: {languages}")
    logging.info(f"Metrics: {metrics}")
    logging.info(f"Result directory: {query_result_dir}")
    logging.info(f"Qrels directory: {eval_args.qrels_dir}")
    logging.info("=" * 70)
    
    # 检查 qrels 文件
    if not check_qrels_files(eval_args.qrels_dir, languages):
        return
    
    # 下载 TREC 评估工具
    logging.info("\nDownloading TREC evaluation tool...")
    script_path = download_evaluation_script('trec_eval')
    logging.info(f"✓ TREC tool path: {script_path}")
    
    # 开始评估
    all_results = {}
    
    for lang in languages:
        logging.info(f"\n{'=' * 70}")
        logging.info(f"Evaluating language: {lang}")
        logging.info("=" * 70)
        
        qrels_path = os.path.join(eval_args.qrels_dir, f"qrels.mldr-v1.0-{lang}-test.tsv")
        logging.info(f"Qrels file: {qrels_path}")
        
        # 查找该语言的所有结果文件
        lang_results = {}
        for filename in os.listdir(query_result_dir):
            if not filename.startswith(f'{lang}_') or not filename.endswith('.txt'):
                continue
            
            query_result_path = os.path.join(query_result_dir, filename)
            if not os.path.isfile(query_result_path):
                continue
            
            logging.info(f"\n  Evaluating file: {filename}")
            logging.info(f"  File path: {query_result_path}")
            
            try:
                result = evaluate(script_path, qrels_path, query_result_path, metrics)
                lang_results[filename] = result
                all_results[f"{lang}_{filename}"] = result
            except Exception as e:
                logging.warning(f"  ✗ Evaluation failed: {e}")
        
        # 显示该语言的评估结果
        if lang_results:
            logging.info(f"\n  {lang} Evaluation Results:")
            for filename, result in lang_results.items():
                logging.info(f"    {filename}:")
                for metric, value in result.items():
                    if isinstance(value, float):
                        logging.info(f"      {metric}: {value:.4f}")
                    else:
                        logging.info(f"      {metric}: {value}")
        else:
            logging.warning(f"  ✗ No result files found for language: {lang}")
    
    # 显示总体结果
    logging.info("\n" + "=" * 70)
    logging.info("Overall Results Summary")
    logging.info("=" * 70)
    
    for result_key, result in all_results.items():
        logging.info(f"\n  {result_key}:")
        for metric, value in result.items():
            if isinstance(value, float):
                logging.info(f"    {metric}: {value:.4f}")
            else:
                logging.info(f"    {metric}: {value}")
    
    logging.info("\n" + "=" * 70)
    logging.info("✓ All evaluations complete!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
 