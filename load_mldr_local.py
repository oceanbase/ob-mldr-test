#!/usr/bin/env python3
"""
使用本地MLDR.py脚本加载数据
"""

import os
import sys
import argparse
from datasets import load_dataset, DownloadConfig

# 导入统一的缓存配置模块
from hf_cache_config import setup_hf_cache, get_cache_dir

# 设置Hugging Face缓存
setup_hf_cache()

# 添加当前目录到Python路径，以便导入MLDR.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

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
        use_auth_token=None,
        force_download=False,
        
    )
    
    try:
        # 使用本地MLDR.py脚本加载数据
        # 这里需要指定本地脚本的路径
        script_path = os.path.join(current_dir, "dataset", "MLDR.py")
        
        if os.path.exists(script_path):
            # 使用本地脚本加载
            if streaming:
                corpus = load_dataset(
                    script_path,  # 本地脚本路径
                    f'corpus-{lang}',  # 配置名称
                    split='corpus',
                    streaming=streaming,
                    trust_remote_code=True
                )
            else:
                corpus = load_dataset(
                    script_path,  # 本地脚本路径
                    f'corpus-{lang}',  # 配置名称
                    split='corpus',
                    download_config=download_config,
                    trust_remote_code=True
                )
            print(f"使用本地MLDR.py成功加载语料库: {lang} (streaming={streaming})")
            return corpus
        else:
           
            print(f" 本地MLDR.py脚本不存在: {script_path}")
            print("回退到远程加载...")
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
        print(f"加载语料库失败: {e}")
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
        use_auth_token=None,
        force_download=False
    )
    
    try:
        # 使用本地MLDR.py脚本加载数据
        script_path = os.path.join(current_dir, "dataset", "MLDR.py")
        
        if os.path.exists(script_path):
            # 使用本地脚本加载
            if streaming:
                dataset = load_dataset(script_path, lang, split='test', streaming=streaming,
                                      trust_remote_code=True)
            else:
                dataset = load_dataset(script_path, lang, split='test',
                                      download_config=download_config,
                                      trust_remote_code=True)
            print(f"使用本地MLDR.py成功加载查询数据: {lang}")
        else:
            # 如果本地脚本不存在，回退到远程加载
            print(f" 本地MLDR.py脚本不存在: {script_path}")
            print("回退到远程加载...")
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
        print(f"加载查询数据失败: {e}")
        raise

def download_all_languages():
    """下载所有语言的语料库和查询数据"""
    print(f"开始下载所有语言的数据，共 {len(available_languages)} 种语言...")
    print(f"语言列表: {', '.join(available_languages)}")
    
    results = {}
    
    for lang in available_languages:
        print(f"\n{'='*50}")
        print(f"正在处理语言: {lang}")
        print(f"{'='*50}")
        
        try:
            # 下载语料库
            print(f"下载语料库: {lang}")
            corpus = load_corpus_local(lang)
            print(f"✓ 语料库下载成功: {lang} (行数: {corpus.num_rows})")
            
            # 下载查询数据
            print(f"下载查询数据: {lang}")
            queries, qids = get_queries_and_qids_local(lang)
            print(f"✓ 查询数据下载成功: {lang} (查询数量: {len(queries)})")
            
            results[lang] = {
                'corpus': corpus,
                'queries': queries,
                'qids': qids,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"✗ 语言 {lang} 下载失败: {e}")
            results[lang] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 输出下载总结
    print(f"\n{'='*60}")
    print("下载总结:")
    print(f"{'='*60}")
    
    success_count = sum(1 for result in results.values() if result['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"成功: {success_count}/{len(available_languages)}")
    print(f"失败: {failed_count}/{len(available_languages)}")
    
    if failed_count > 0:
        print("\n失败的语言:")
        for lang, result in results.items():
            if result['status'] == 'failed':
                print(f"  - {lang}: {result['error']}")
    
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
        print(f"开始下载语言: {lang}")
        
        try:
            if not args.queries_only:
                print(f"下载语料库: {lang}")
                corpus = load_corpus_local(lang)
                print(f"语料库行数: {corpus.num_rows}")
                print(f"示例文档ID: {corpus['docid'][:3]}")
                print(f"语料库缓存路径: {get_cache_dir()}")
            
            if not args.corpus_only:
                print(f"下载查询数据: {lang}")
                queries, qids = get_queries_and_qids_local(lang)
                print(f"查询数量: {len(queries)}")
                print(f"示例查询: {queries[:3]}")
                print(f"查询数据缓存路径: {get_cache_dir()}")
                
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)
    else:
        # 默认行为：测试西班牙语
        print("测试本地MLDR.py加载...")
        print("使用 --help 查看可用选项")
        print("使用 --all 下载所有语言")
        print("使用 --lang <语言代码> 下载指定语言")
        print()

        # 测试语料库加载
        try:
            corpus = load_corpus_local('es')
            print(f"语料库行数: {corpus.num_rows}")
            print(f"示例文档ID: {corpus['docid'][:3]}")
            # 输出语料库存储路径
            print(f"语料库缓存路径: {get_cache_dir()}")  # 打印缓存目录
        except Exception as e:
            print(f"语料库加载失败: {e}")

        # 测试查询数据加载
        try:
            queries, qids = get_queries_and_qids_local('es')
            print(f"查询数量: {len(queries)}")
            print(f"示例查询: {queries[:3]}")
            # 输出查询数据缓存路径
            print(f"查询数据缓存路径: {get_cache_dir()}")  # 打印缓存目录
        except Exception as e:
            print(f"查询数据加载失败: {e}")

if __name__ == "__main__":
    main()