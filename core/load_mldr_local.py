#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地MLDR.py脚本加载数据
"""

import os
import sys
from .hf_cache_config import get_cache_dir
import argparse
from datasets import load_dataset, DownloadConfig
import logging

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # core/
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.insert(0, project_root)

# 定义可用的语言列表
available_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']

def load_corpus_local(lang: str, streaming: bool = False):
    """使用本地MLDR.py加载语料库"""
    # 获取缓存目录
    custom_cache_dir = get_cache_dir()
    
    # 创建缓存目录
    os.makedirs(custom_cache_dir, exist_ok=True)
    
    # 设置下载配置
    download_config = DownloadConfig(
        resume_download=True,
        cache_dir=custom_cache_dir,
        local_files_only=False,
        token=None,
        force_download=False,
    )
    
    try:
        # 使用项目根目录
        script_path = os.path.join(project_root, "dataset", "MLDR.py")
        
        if os.path.exists(script_path):
            # 使用本地脚本加载
            if streaming:
                corpus = load_dataset(
                    script_path,  # 本地脚本路径
                    f'corpus-{lang}',  # 配置名称
                    split='corpus',
                    streaming=streaming,
                    trust_remote_code=True,
                    token=None
                )
            else:
                corpus = load_dataset(
                    script_path,  # 本地脚本路径
                    f'corpus-{lang}',  # 配置名称
                    split='corpus',
                    download_config=download_config,
                    trust_remote_code=True
                )
            logging.debug(f"使用本地MLDR.py成功加载语料库: {lang} (streaming={streaming})")
            return corpus
        else:
            logging.debug(f"⚠ 本地MLDR.py脚本不存在: {script_path}")
            logging.debug("回退到远程加载...")
            if streaming:
                corpus = load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus',
                                     streaming=streaming,
                                     trust_remote_code=True)
            else:
                corpus = load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus',
                                     download_config=download_config,
                                     trust_remote_code=True)
            return corpus
            
    except Exception as e:
        logging.error(f"加载语料库失败: {e}")
        raise

def get_queries_and_qids_local(lang: str, streaming: bool = False):
    """使用本地MLDR.py加载查询数据"""
    # 获取缓存目录
    custom_cache_dir = get_cache_dir()
    
    # 创建缓存目录
    os.makedirs(custom_cache_dir, exist_ok=True)
    
    # 设置下载配置
    download_config = DownloadConfig(
        resume_download=True,
        cache_dir=custom_cache_dir,
        local_files_only=False,
        # use_auth_token=None,
        token=None,
        force_download=False
    )
    
    try:
        script_path = os.path.join(project_root, "dataset", "MLDR.py")
        
        if os.path.exists(script_path):
            # 使用本地脚本加载
            if streaming:
                dataset = load_dataset(script_path, lang, split='test', streaming=streaming,
                                      trust_remote_code=True)
            else:
                dataset = load_dataset(script_path, lang, split='test',
                                      download_config=download_config,
                                      trust_remote_code=True)
            logging.debug(f"使用本地MLDR.py成功加载查询数据: {lang}")
        else:
            # 如果本地脚本不存在，回退到远程加载
            logging.debug(f"⚠ 本地MLDR.py脚本不存在: {script_path}")
            logging.debug("回退到远程加载...")
            if streaming:
                dataset = load_dataset('Shitao/MLDR', lang, split='test', streaming=streaming,
                                      trust_remote_code=True)
            else:
                dataset = load_dataset('Shitao/MLDR', lang, split='test',
                                      download_config=download_config,
                                      trust_remote_code=True)
        
        queries = []
        qids = []
        for data in dataset:
            qids.append(data['query_id'])
            queries.append(data['query'])
        return queries, qids
        
    except Exception as e:
        logging.error(f"加载查询数据失败: {e}")
        raise

def download_all_languages():
    """下载所有语言的语料库和查询数据"""
    logging.debug(f"开始下载所有语言的数据，共 {len(available_languages)} 种语言...")
    logging.debug(f"语言列表: {', '.join(available_languages)}")
    
    results = {}
    
    for lang in available_languages:
        logging.debug(f"\n{'='*50}")
        logging.debug(f"正在处理语言: {lang}")
        logging.debug(f"{'='*50}")
        
        try:
            # 下载语料库
            logging.debug(f"下载语料库: {lang}")
            corpus = load_corpus_local(lang)
            logging.debug(f"✓ 语料库下载成功: {lang} (行数: {corpus.num_rows})")
            
            # 下载查询数据
            logging.debug(f"下载查询数据: {lang}")
            queries, qids = get_queries_and_qids_local(lang)
            logging.debug(f"✓ 查询数据下载成功: {lang} (查询数量: {len(queries)})")
            
            results[lang] = {
                'corpus': corpus,
                'queries': queries,
                'qids': qids,
                'status': 'success'
            }
            
        except Exception as e:
            logging.warning(f"✗ 语言 {lang} 下载失败: {e}")
            results[lang] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 输出下载总结
    logging.debug(f"\n{'='*60}")
    logging.debug("下载总结:")
    logging.debug(f"{'='*60}")
    
    success_count = sum(1 for result in results.values() if result['status'] == 'success')
    failed_count = len(results) - success_count
    
    logging.debug(f"成功: {success_count}/{len(available_languages)}")
    logging.debug(f"失败: {failed_count}/{len(available_languages)}")
    
    if failed_count > 0:
        logging.warning("\n失败的语言:")
        for lang, result in results.items():
            if result['status'] == 'failed':
                logging.warning(f"  - {lang}: {result['error']}")
    
    return results

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='MLDR数据加载工具')
    parser.add_argument('--all', action='store_true', 
                       help='下载所有语言的数据')
    parser.add_argument('--lang', type=str, choices=available_languages,
                       help='指定要下载的语言')
    parser.add_argument('--corpus-only', action='store_true',
                       help='仅下载语料库，不下载查询数据')
    parser.add_argument('--queries-only', action='store_true',
                       help='仅下载查询数据，不下载语料库')
    
    args = parser.parse_args()
    
    if args.all:
        # 下载所有语言
        download_all_languages()
    elif args.lang:
        # 下载指定语言
        lang = args.lang
        logging.info(f"开始下载语言: {lang}")
        
        try:
            if not args.queries_only:
                logging.info(f"下载语料库: {lang}")
                corpus = load_corpus_local(lang)
                logging.info(f"语料库行数: {corpus.num_rows}")
                logging.info(f"示例文档ID: {corpus['docid'][:3]}")
                logging.info(f"语料库缓存路径: {get_cache_dir()}")
            
            if not args.corpus_only:
                logging.info(f"下载查询数据: {lang}")
                queries, qids = get_queries_and_qids_local(lang)
                logging.info(f"查询数量: {len(queries)}")
                logging.info(f"示例查询: {queries[:3]}")
                logging.info(f"查询数据缓存路径: {get_cache_dir()}")
                
        except Exception as e:
            logging.error(f"下载失败: {e}")
            sys.exit(1)
    else:
        # 默认行为：测试西班牙语
        logging.info("测试本地MLDR.py加载...")
        logging.info("使用 --help 查看可用选项")
        logging.info("使用 --all 下载所有语言")
        logging.info("使用 --lang <语言代码> 下载指定语言")

        # 测试语料库加载
        try:
            corpus = load_corpus_local('es')
            logging.info(f"语料库行数: {corpus.num_rows}")
            logging.info(f"示例文档ID: {corpus['docid'][:3]}")
            # 输出语料库存储路径
            logging.info(f"语料库缓存路径: {get_cache_dir()}")  # 打印缓存目录
        except Exception as e:
            logging.error(f"语料库加载失败: {e}")

        # 测试查询数据加载
        try:
            queries, qids = get_queries_and_qids_local('es')
            logging.info(f"查询数量: {len(queries)}")
            logging.info(f"示例查询: {queries[:3]}")
            # 输出查询数据缓存路径
            logging.info(f"查询数据缓存路径: {get_cache_dir()}")  # 打印缓存目录
        except Exception as e:
            logging.error(f"查询数据加载失败: {e}")

if __name__ == "__main__":
    main()