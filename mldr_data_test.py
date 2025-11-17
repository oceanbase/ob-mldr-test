#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 数据集测试集成类
整合数据插入、搜索和评估三个步骤
这是给选手的主要测试文件，只需运行这一个文件即可完成所有测试
"""

import os
import sys
import time
import logging
import tempfile
import platform
import subprocess
from typing import Tuple, Optional

# 导入其他模块的功能
from test_insert_fast import test_simple_insert
from get_search_rrf_oceanbase import OceanBaseClientForSearch, ModelArgs
from evaluate_results_oceanbase import evaluate, check_qrels_files, map_metric
from pyserini.util import download_evaluation_script

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLDRDataTest:
    """MLDR 数据集测试类"""
    
    def __init__(self, 
                 mldr_data_test_dir: str,
                 lang: str = 'en',
                 query_types: str = 'bm25',
                 db_port: int = 2881,
                 query_result_dir: Optional[str] = None):
        """
        初始化 MLDR 测试
        
        Args:
            mldr_data_test_dir: mldr-data-test 目录路径
            lang: 测试语言，默认 'en'
            query_types: 查询类型，默认 'bm25'
            db_port: 数据库端口，默认 2881
            query_result_dir: 查询结果保存目录，如果为 None 则使用临时目录
        """
        self.mldr_data_test_dir = mldr_data_test_dir
        self.lang = lang
        self.query_types = query_types
        self.db_port = db_port
        self.query_result_dir = query_result_dir or os.path.join(tempfile.gettempdir(), 'mldr_query_results')
        
        # 确保结果目录存在
        os.makedirs(self.query_result_dir, exist_ok=True)
        
        # 测试结果
        self.recall_at_10: float = 0.0
        self.qps: float = 0.0
        self.total_queries: int = 0
        self.total_time: float = 0.0
        self.all_runs_passed: bool = False  # 是否所有测试都达到阈值
        
        logger.info(f'MLDRDataTest initialized: lang={lang}, query_types={query_types}, db_port={db_port}')
        logger.info(f'Query result directory: {self.query_result_dir}')
    
    def step1_insert_data(self) -> bool:
        """
        步骤1: 插入数据和创建索引
        
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始插入数据: lang={self.lang}")
            test_simple_insert(self.lang)
            logger.info("✓ Data insertion and index creation complete")
            return True
        except Exception as e:
            error_msg = f"step1 insert data failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def step2_search(self) -> Tuple[float, int]:
        """
        步骤2: 执行搜索
        
        Returns:
            Tuple[float, int]: (总查询时间, 总查询数)
        """
        try:
            logger.info(f"开始搜索: lang={self.lang}, query_types={self.query_types}")
            
            # 创建搜索客户端
            oceanbase_client = OceanBaseClientForSearch(with_colbert=False)
            
            # 创建模型参数
            model_args = ModelArgs()
            
            # 执行搜索
            query_types_list = [self.query_types] if isinstance(self.query_types, str) else self.query_types
            oceanbase_client.main(
                languages=[self.lang],
                query_types=query_types_list,
                model_args=model_args,
                save_dir=self.query_result_dir
            )
            
            # 关闭连接
            if oceanbase_client.connection:
                oceanbase_client.connection.close()
            
            # 释放客户端对象
            del oceanbase_client
            import gc
            gc.collect()
            
            # 从结果文件中统计查询数
            expected_file = os.path.join(self.query_result_dir, f"{self.lang}_{self.query_types}.txt")
            total_queries = 0
            if os.path.exists(expected_file):
                try:
                    with open(expected_file, 'r') as f:
                        lines = f.readlines()
                        # TREC 格式，每行一个查询结果，通过查询ID判断查询数
                        query_ids = set()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                query_ids.add(parts[0])
                        total_queries = len(query_ids)
                        logger.info(f'Found total queries from result file: {total_queries}')
                except Exception as e:
                    logger.warning(f'Failed to count queries from result file: {e}')
            
            logger.info("✓ Search complete")
            return 0.0, total_queries
            
        except Exception as e:
            error_msg = f"step2 search failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def step3_evaluate(self) -> float:
        """
        步骤3: 评估搜索结果
        
        Returns:
            float: recall@10 值
        """
        logger.info("=" * 60)
        logger.info("step3: Evaluate search results")
        logger.info("=" * 60)
        
        try:
            # 检查qrels文件
            qrels_dir = './qrels'
            if not check_qrels_files(qrels_dir, [self.lang]):
                error_msg = f"step3 evaluate failed: qrels file not found for lang={self.lang}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 下载TREC评估工具
            logger.info("下载TREC评估工具...")
            script_path = download_evaluation_script('trec_eval')
            logger.info(f"TREC评估工具路径: {script_path}")
            
            # 评估结果文件
            query_result_path = os.path.join(self.query_result_dir, f"{self.lang}_{self.query_types}.txt")
            if not os.path.exists(query_result_path):
                error_msg = f"step3 evaluate failed: result file not found: {query_result_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            qrels_path = os.path.join(qrels_dir, f"qrels.mldr-v1.0-{self.lang}-test-filtered.tsv")
            metrics = ['recall@10']
            
            # 执行评估
            result = evaluate(script_path, qrels_path, query_result_path, metrics)
            
            # 提取 recall@10
            recall_at_10 = result.get('recall@10', 0.0)
            if isinstance(recall_at_10, str):
                # 如果解析失败，尝试从字符串中提取
                import re
                match = re.search(r'(\d+\.\d+)', recall_at_10)
                if match:
                    recall_at_10 = float(match.group(1))
                else:
                    recall_at_10 = 0.0
            
            if recall_at_10 == 0.0:
                error_msg = "step3 evaluate failed: cannot extract recall@10 from result"
                logger.error(error_msg)
                logger.error(f"Result: {result}")
                raise RuntimeError(error_msg)
            
            logger.info(f"✓ Evaluation complete: recall@10={recall_at_10:.4f}")
            return recall_at_10
            
        except Exception as e:
            error_msg = f"step3 evaluate failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def run_all(self) -> Tuple[float, float]:
        """
        运行完整的测试流程
        1. 一次预热查询（不算成绩）
        2. 3次正式查询，取平均值
        
        Returns:
            Tuple[float, float]: (recall@10, QPS)
        """
        logger.info("=" * 60)
        logger.info("Starting MLDR data test")
        logger.info("=" * 60)
        
        try:
            # 步骤1: 插入数据
            self.step1_insert_data()
            
            # 预热：执行一次搜索和评估（不算成绩）
            logger.info("=" * 60)
            logger.info("Warm-up: Running one search and evaluation (not counted)")
            logger.info("=" * 60)
            self._run_single_search_and_eval(warmup=True)
            logger.info("✓ Warm-up complete")
            
            # 正式测试：执行3次搜索和评估，取平均值
            logger.info("=" * 60)
            logger.info("Formal test: Running 3 searches and evaluations, taking average")
            logger.info("=" * 60)
            
            recall_list = []
            qps_list = []
            recall_threshold = 0.95
            
            for i in range(3):
                logger.info(f"\n--- Formal test run {i+1}/3 ---")
                recall, qps = self._run_single_search_and_eval(warmup=False)
                recall_list.append(recall)
                qps_list.append(qps)
                logger.info(f"Run {i+1}/3: recall@10={recall:.4f}, QPS={qps:.2f}")
                
                # 检查每次测试是否都达到阈值
                if recall < recall_threshold:
                    logger.warning(f"Run {i+1}/3: recall@10={recall:.4f} < {recall_threshold}, this run does not meet threshold")
            
            # 计算平均值
            self.recall_at_10 = sum(recall_list) / len(recall_list)
            self.qps = sum(qps_list) / len(qps_list)
            
            # 检查是否所有测试都达到阈值
            all_passed = all(r >= recall_threshold for r in recall_list)
            
            logger.info("=" * 60)
            logger.info("MLDR data test complete")
            logger.info(f"Average recall@10: {self.recall_at_10:.4f} (from {len(recall_list)} runs)")
            logger.info(f"Average QPS: {self.qps:.2f} (from {len(qps_list)} runs)")
            logger.info(f"Individual results:")
            for i, (r, q) in enumerate(zip(recall_list, qps_list), 1):
                status = "✓" if r >= recall_threshold else "✗"
                logger.info(f"  Run {i}: recall@10={r:.4f}, QPS={q:.2f} {status}")
            
            if all_passed:
                logger.info(f"✓ All {len(recall_list)} runs meet recall@10 >= {recall_threshold} threshold")
            else:
                failed_runs = [i+1 for i, r in enumerate(recall_list) if r < recall_threshold]
                logger.warning(f"✗ Some runs do not meet recall@10 >= {recall_threshold} threshold: runs {failed_runs}")
            
            logger.info("=" * 60)
            
            # 保存是否所有测试都通过的信息（用于后续判断）
            self.all_runs_passed = all_passed
            
            return self.recall_at_10, self.qps
            
        except Exception as e:
            logger.error(f"MLDR data test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _run_single_search_and_eval(self, warmup: bool = False) -> Tuple[float, float]:
        """
        执行一次搜索和评估
        
        Args:
            warmup: 是否为预热（预热时不清理结果文件，正式测试时需要清理）
        
        Returns:
            Tuple[float, float]: (recall@10, QPS)
        """
        # 如果不是预热，清理之前的结果文件
        if not warmup:
            expected_file = os.path.join(self.query_result_dir, f"{self.lang}_{self.query_types}.txt")
            if os.path.exists(expected_file):
                try:
                    os.remove(expected_file)
                    logger.info(f"Cleaned previous result file: {expected_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean result file: {e}")
        
        # 执行搜索
        search_start_time = time.time()
        try:
            logger.info(f"开始搜索: lang={self.lang}, query_types={self.query_types}")
            
            # 创建搜索客户端
            oceanbase_client = OceanBaseClientForSearch(with_colbert=False)
            
            # 创建模型参数
            model_args = ModelArgs()
            
            # 执行搜索
            query_types_list = [self.query_types] if isinstance(self.query_types, str) else self.query_types
            oceanbase_client.main(
                languages=[self.lang],
                query_types=query_types_list,
                model_args=model_args,
                save_dir=self.query_result_dir
            )
            
            # 关闭连接
            if oceanbase_client.connection:
                oceanbase_client.connection.close()
            
            # 释放客户端对象
            del oceanbase_client
            import gc
            gc.collect()
            
            search_end_time = time.time()
            search_duration = search_end_time - search_start_time
            
            # 从结果文件中统计查询数并计算QPS
            expected_file = os.path.join(self.query_result_dir, f"{self.lang}_{self.query_types}.txt")
            total_queries = 0
            if os.path.exists(expected_file):
                try:
                    with open(expected_file, 'r') as f:
                        lines = f.readlines()
                        # TREC 格式，每行一个查询结果，通过查询ID判断查询数
                        query_ids = set()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                query_ids.add(parts[0])
                        total_queries = len(query_ids)
                        logger.info(f'Found total queries from result file: {total_queries}')
                except Exception as e:
                    logger.warning(f'Failed to count queries from result file: {e}')
            
            # 计算 QPS
            if total_queries > 0 and search_duration > 0:
                qps = total_queries / search_duration
                logger.info(f"Search QPS: {qps:.2f}, total queries: {total_queries}, duration: {search_duration:.2f}s")
            else:
                qps = 0.0
                logger.warning(f"Cannot calculate QPS: total_queries={total_queries}, duration={search_duration:.2f}s")
            
        except Exception as e:
            error_msg = f"search failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # 执行评估
        recall_at_10 = self.step3_evaluate()
        logger.info(f"Evaluation recall@10: {recall_at_10:.4f}")
        
        return recall_at_10, qps


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MLDR数据集测试主程序')
    parser.add_argument('--lang', type=str, default='en', 
                       help='测试语言，默认 en')
    parser.add_argument('--query_types', type=str, default='bm25',
                       help='查询类型，默认 bm25')
    parser.add_argument('--db_port', type=int, default=2881,
                       help='数据库端口，默认 2881')
    parser.add_argument('--query_result_dir', type=str, default=None,
                       help='查询结果保存目录，如果为 None 则使用临时目录')
    
    args = parser.parse_args()
    
    # 获取当前脚本所在目录作为 mldr_data_test_dir
    mldr_data_test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建测试实例
    test = MLDRDataTest(
        mldr_data_test_dir=mldr_data_test_dir,
        lang=args.lang,
        query_types=args.query_types,
        db_port=args.db_port,
        query_result_dir=args.query_result_dir
    )
    
    # 运行测试
    try:
        recall, qps = test.run_all()
        print(f"\n{'='*60}")
        print(f"最终结果:")
        print(f"  recall@10: {recall:.4f}")
        print(f"  QPS: {qps:.2f}")
        print(f"{'='*60}")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

