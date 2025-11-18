#!/usr/bin/env python3
"""
OceanBase版本的搜索结果评估脚本
使用TREC评估工具评估搜索质量
"""

import os

# 导入统一的缓存配置模块
from hf_cache_config import setup_hf_cache, get_cache_dir

# 设置Hugging Face缓存
# setup_hf_cache()

import platform
import subprocess
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pyserini.util import download_evaluation_script
from mldr_common_tools import check_languages


# def setup_mirror():
#     """设置HF-Mirror镜像站和缓存"""
#     setup_hf_cache()

# setup_mirror()


@dataclass
class EvalArgs:
    languages: str = field(default="en", metadata={
        'help': 'Languages to evaluate. Available languages: ar de en es fr hi it ja ko pt ru th zh', "nargs": "+"})
    query_result_dave_dir: str = field(default='', metadata={'help': 'Dir to save query result. If not specified, will auto-generate based on languages.'})
    qrels_dir: str = field(default='./qrels', metadata={'help': 'Dir to qrels.'})
    metrics: str = field(default="ndcg@10",
                         metadata={'help': 'Metrics to evaluate. Available metrics: ndcg@k, recall@k', "nargs": "+"})


all_query_types = ['bm25', 'dense', 'sparse']


def map_metric(metric: str):
    """映射评估指标到TREC格式"""
    metric, k = metric.split('@')
    if metric.lower() == 'ndcg':
        return k, f'ndcg_cut.{k}'
    elif metric.lower() == 'recall':
        return k, f'recall.{k}'
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate(script_path, qrels_path, query_result_path, metrics: list):
    """使用TREC评估工具评估搜索结果"""
    cmd_prefix = ['java', '-jar', script_path]
    #########################################################
    # 将距离分数转换为相似度分数（取负值）
    #########################################################

    # 创建转换后的查询结果文件（将距离分数转换为相似度分数）l2
    # import tempfile
    # import os
    
    # converted_result_path = query_result_path + '.converted'
    # with open(query_result_path, 'r') as f_in, open(converted_result_path, 'w') as f_out:
    #     for line in f_in:
    #         parts = line.strip().split()
    #         if len(parts) >= 5:
    #             # 将距离分数转换为相似度分数（取负值）
    #             parts[4] = str(-float(parts[4]))
    #             f_out.write(' '.join(parts) + '\n')
    #         else:
    #             f_out.write(line)
    #########################################################


    results = {}
    for metric in metrics:
        k, mapped_metric = map_metric(metric)
        args = ['-c', '-M', str(k), '-m', mapped_metric, qrels_path, query_result_path]
        #args = ['-c', '-M', str(k), '-m', mapped_metric, qrels_path, converted_result_path]
        cmd = cmd_prefix + args

        print(f'运行评估命令: {" ".join(cmd)}')
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        stdout, stderr = process.communicate()
        
        if stderr:
            print(f" 警告信息: {stderr.decode('utf-8')}")
        
        result_str = stdout.decode("utf-8")
        try:
            results[metric] = float(result_str.split(' ')[-1].split('\t')[-1])
            print(f"{metric}: {results[metric]:.4f}")
        except Exception as e:
            print(f"解析结果失败: {e}")
            print(f"原始输出: {result_str}")
            results[metric] = result_str
    
    # 清理临时文件
    #os.remove(converted_result_path)
    
    return results


def check_qrels_files(qrels_dir: str, languages: list):
    """检查qrels文件是否存在"""
    missing_files = []
    for lang in languages:
        qrels_path = os.path.join(qrels_dir, f"qrels.mldr-v1.0-{lang}-test-filtered.tsv")
        if not os.path.exists(qrels_path):
            missing_files.append(qrels_path)
    
    if missing_files:
        print("缺少以下qrels文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请下载qrels文件:")
        print("   wget https://hf-mirror.com/datasets/Shitao/MLDR/resolve/main/qrels/qrels.mldr-v1.0-en-test.tsv")
        print("   cp qrels.mldr-v1.0-en-test.tsv ./qrels/")
        return False
    
    return True


def main():
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    # 自动生成存储目录路径
    if not eval_args.query_result_dave_dir:
        from mldr_common_tools import get_default_query_result_dir
        query_result_dave_dir = get_default_query_result_dir(eval_args.languages)
        print(f"未指定结果目录，自动生成: {query_result_dave_dir}")
    else:
        query_result_dave_dir = eval_args.query_result_dave_dir
        print(f"使用指定的结果目录: {query_result_dave_dir}")
    
    languages = check_languages(eval_args.languages)
    metrics = eval_args.metrics if isinstance(eval_args.metrics, list) else [eval_args.metrics]
    
    print("开始评估OceanBase搜索结果")
    print("=" * 60)
    print(f"评估语言: {languages}")
    print(f"评估指标: {metrics}")
    print(f"结果目录: {query_result_dave_dir}")
    print(f"Qrels目录: {eval_args.qrels_dir}")
    print("=" * 60)
    
    # 检查qrels文件
    if not check_qrels_files(eval_args.qrels_dir, languages):
        return
    
    # 下载TREC评估工具
    print("下载TREC评估工具...")
    script_path = download_evaluation_script('trec_eval')
    print(f"TREC评估工具路径: {script_path}")
    
    # 开始评估
    all_results = {}
    
    for lang in languages:
        print(f"\n评估语言: {lang}")
        print("-" * 40)
        
        qrels_path = os.path.join(eval_args.qrels_dir, f"qrels.mldr-v1.0-{lang}-test-filtered.tsv")
        print(f" Qrels文件: {qrels_path}")
        
        # 查找该语言的所有结果文件
        lang_results = {}
        for filename in os.listdir(query_result_dave_dir):
            if not filename.startswith(f'{lang}_') or not filename.endswith('.txt'):
                continue
            
            query_result_path = os.path.join(query_result_dave_dir, filename)
            if not os.path.isfile(query_result_path):
                continue
            
            print(f"\n 评估文件: {filename}")
            print(f"文件路径: {query_result_path}")
            
            try:
                result = evaluate(script_path, qrels_path, query_result_path, metrics)
                lang_results[filename] = result
                all_results[f"{lang}_{filename}"] = result
            except Exception as e:
                print(f"评估失败: {e}")
        
        # 显示该语言的评估结果
        if lang_results:
            print(f"\n {lang} 语言评估结果:")
            for filename, result in lang_results.items():
                print(f"  {filename}:")
                for metric, value in result.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
    
    # 显示总体结果
    print("\n" + "=" * 60)
    print("评估完成！总体结果:")
    print("=" * 60)
    
    for result_key, result in all_results.items():
        print(f"\n {result_key}:")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n" + "=" * 60)
    print("所有评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main() 