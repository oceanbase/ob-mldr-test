#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLDR 数据集测试运行器 - 主入口文件

整合数据插入、搜索和评估的完整测试流程。
支持 OceanBase 后端和多种查询类型。

使用示例:
    # OceanBase 全文检索测试
    python mldr_test_runner.py --lang en --backend oceanbase --query-type bm25
    
    # OceanBase 混合检索测试
    python mldr_test_runner.py --lang en --backend oceanbase --query-type hybrid_dense_bm25
    
    # 多次测试取平均值
    python mldr_test_runner.py --lang en --query-type bm25 --runs 3
"""

import os
import sys
import time
import logging
import tempfile
import argparse
from typing import Tuple, Optional, List, Dict, Any

logging.root.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler],
    force=True
)
logger = logging.getLogger(__name__)
from config_loader import (
    load_oceanbase_hybrid_config,
    set_global_config
)
from database import (
    insert_data_oceanbase,
)
from search import (
    OceanBaseClientForHybridSearch,
    QueryType
)
from evaluation import (
    evaluate_search_results,
    check_qrels_files
)
from core import get_queries_and_qids


class MLDRTestRunner:
    """MLDR 数据集测试运行器"""
    
    QUERY_TYPE_MAP = {
        'bm25': QueryType.BM25,
        'dense': QueryType.DENSE,
        'sparse': QueryType.SPARSE,
        'colbert': QueryType.COLBERT,
        'hybrid_dense_bm25': QueryType.HYBRID_DENSE_BM25,
        'fusion_rrf': QueryType.FUSION_RRF,
        'fusion_weighted': QueryType.FUSION_WEIGHTED,
    }
    
    def __init__(
        self,
        lang: str = 'en',
        backend: str = 'oceanbase',
        query_type: str = 'bm25',
        data_count: int = 200000,
        skip_insert: bool = False,
        result_dir: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """
        初始化测试运行器
        
        Args:
            lang: 测试语言 (en, zh, etc.)
            backend: 后端数据库（目前仅支持 'oceanbase'）
            query_type: 查询类型 ('bm25', 'dense', 'hybrid_dense_bm25', etc.)
            data_count: 要插入的数据量
            skip_insert: 是否跳过数据插入步骤
            result_dir: 结果保存目录
            config_file: 配置文件路径
            **kwargs: 其他配置参数
        """
        self.lang = lang
        self.backend = backend.lower()
        self.query_type = query_type.lower()
        self.data_count = data_count
        self.skip_insert = skip_insert
        self.result_dir = result_dir or os.path.join(
            tempfile.gettempdir(),
            f'mldr_test_results_{int(time.time())}'
        )
        self.config_file = config_file
        self.kwargs = kwargs

        if self.config_file:
            set_global_config(self.config_file)
            logger.info(f"使用配置文件: {self.config_file}")

        os.makedirs(self.result_dir, exist_ok=True)
        self._validate_params()
        self.recall_at_10: float = 0.0
        self.ndcg_at_10: float = 0.0
        self.avg_query_time_ms: float = 0.0

    def _validate_params(self):
        """验证参数有效性"""
        if self.backend not in ['oceanbase']:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'oceanbase'")

        if self.query_type not in self.QUERY_TYPE_MAP:
            valid_types = ', '.join(self.QUERY_TYPE_MAP.keys())
            raise ValueError(f"Invalid query_type: {self.query_type}. Must be one of: {valid_types}")

        if self.query_type in ['fusion_rrf', 'fusion_weighted']:
            if self.backend != 'oceanbase':
                raise ValueError(f"Query type '{self.query_type}' is only supported with OceanBase backend")

    def step1_insert_data(self) -> bool:
        """步骤1: 插入数据和创建索引"""
        if self.skip_insert:
            logger.info("=" * 70)
            logger.info("Step 1: Insert data [SKIPPED]")
            logger.info("=" * 70)
            return True
        
        logger.info("=" * 70)
        logger.info("Step 1: Insert data and create indexes")
        logger.info("=" * 70)
        
        try:
            if self.backend == 'oceanbase':
                logger.info(f"Inserting data into OceanBase (count={self.data_count})...")

                from config_loader import load_oceanbase_insert_config
                config = load_oceanbase_insert_config(self.config_file)
                config.test_count = self.data_count
                insert_data_oceanbase(lang=self.lang, config=config)
                
        except Exception as e:
            logger.error(f"✗ Step 1 failed: {e}")
            raise

    def step2_search(self) -> Tuple[float, int]:
        """步骤2: 执行搜索"""
        logger.info("=" * 70)
        logger.info("Step 2: Execute search queries")
        logger.info("=" * 70)
        
        try:
            queries, query_ids = get_queries_and_qids(self.lang)
            total_queries = len(queries)

            logger.info(f"Loaded {total_queries} queries for language: {self.lang}")

            result_file = os.path.join(self.result_dir, f"{self.lang}_{self.query_type}.txt")
            logger.info(f"Starting search (type={self.query_type})...")

            if self.backend == 'oceanbase':
                avg_sql_time_ms = self._execute_oceanbase_search(queries, query_ids, result_file)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            logger.info(f"✓ Search complete")
            logger.info(f"  Total queries: {total_queries}")
            logger.info(f"  Average SQL execution time per query: {avg_sql_time_ms:.2f}ms")
            logger.info(f"  Results saved to: {result_file}")

            return avg_sql_time_ms, total_queries

        except Exception as e:
            logger.error(f"✗ Step 2 failed: {e}")
            raise
    
    def _execute_oceanbase_search(self, queries: list, query_ids: list, result_file: str) -> float:
        """执行 OceanBase 搜索"""
        from core.embeddings import ModelArgs

        model_args = ModelArgs()
        with_colbert = self.query_type in ['colbert']
        config = load_oceanbase_hybrid_config(self.config_file)
        client = OceanBaseClientForHybridSearch(
            with_colbert=with_colbert,
            config=config
        )

        avg_times = client.main(
            languages=[self.lang],
            query_types=[self.query_type],
            model_args=model_args,
            save_dir=self.result_dir
        )

        key = f"{self.lang}_{self.query_type}"
        avg_sql_time_ms = avg_times.get(key, 0.0)

        if hasattr(client, 'connection') and client.connection:
            client.connection.close()

        return avg_sql_time_ms

    def step3_evaluate(self) -> dict:
        """步骤3: 评估搜索结果"""
        logger.info("=" * 70)
        logger.info("Step 3: Evaluate search results")
        logger.info("=" * 70)
        
        try:
            qrels_dir = './qrels'
            if not check_qrels_files(qrels_dir, [self.lang], auto_download=True):
                raise FileNotFoundError(f"Qrels file not found for language: {self.lang}")

            result_file = os.path.join(self.result_dir, f"{self.lang}_{self.query_type}.txt")
            qrels_file = os.path.join(qrels_dir, f"qrels.mldr-v1.0-{self.lang}-test.tsv")

            if not os.path.exists(result_file):
                raise FileNotFoundError(f"Result file not found: {result_file}")
            
            metrics = evaluate_search_results(
                qrels_file=qrels_file,
                results_file=result_file,
                metrics=['recall@10', 'ndcg@10']
            )

            ndcg_at_10 = metrics.get('ndcg@10', 0.0)
            recall_at_10 = metrics.get('recall@10', 0.0)
            
            logger.info(f"✓ Evaluation complete")
            logger.info(f"  Recall@10: {recall_at_10:.4f}")
            logger.info(f"  NDCG@10: {ndcg_at_10:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"✗ Step 3 failed: {e}")
            raise
    
    
    def run(self, warmup: bool = True, num_runs: int = 1):
        """运行完整的测试流程"""
        logger.info("=" * 70)
        logger.info("Starting MLDR Test")
        logger.info("=" * 70)
        
        try:
            self.step1_insert_data()

            if warmup:
                logger.info("\n" + "=" * 70)
                logger.info("Warm-up run (not counted in results)")
                logger.info("=" * 70)
                try:
                    self.step2_search()
                    logger.info("✓ Warm-up complete")
                except Exception as e:
                    logger.warning(f"Warm-up failed (continuing anyway): {e}")

            recall_list = []
            avg_time_list = []
            ndcg_list = []

            for i in range(num_runs):
                if num_runs > 1:
                    logger.info("\n" + "=" * 70)
                    logger.info(f"Test Run {i+1}/{num_runs}")
                    logger.info("=" * 70)

                avg_time_ms, total_queries = self.step2_search()
                metrics = self.step3_evaluate()

                recall = metrics.get('recall@10', 0.0)
                ndcg = metrics.get('ndcg@10', 0.0)
                recall_list.append(recall)
                avg_time_list.append(avg_time_ms)
                ndcg_list.append(ndcg)
                
                if num_runs > 1:
                    logger.info(f"\nRun {i+1} Results:")
                    logger.info(f"  Recall@10: {recall:.4f}")
                    logger.info(f"  NDCG@10: {ndcg:.4f}")
                    logger.info(f"  Avg query time: {avg_time_ms:.2f}ms")

            self.recall_at_10 = sum(recall_list) / len(recall_list)
            self.ndcg_at_10 = sum(ndcg_list) / len(ndcg_list)
            self.avg_query_time_ms = sum(avg_time_list) / len(avg_time_list)

            logger.info("\n" + "=" * 70)
            logger.info("FINAL RESULTS")
            logger.info("=" * 70)
            logger.info(f"Test Configuration:")
            logger.info(f"  Language: {self.lang}")
            logger.info(f"  Backend: {self.backend}")
            logger.info(f"  Query type: {self.query_type}")
            logger.info(f"  Data count: {self.data_count}")
            logger.info(f"  Number of runs: {num_runs}")
            logger.info(f"\nAverage Performance:")
            logger.info(f"  Recall@10: {self.recall_at_10:.4f}")
            logger.info(f"  NDCG@10: {self.ndcg_at_10:.4f}")
            logger.info(f"  Avg query time: {self.avg_query_time_ms:.2f}ms")
            
            if num_runs > 1:
                logger.info(f"\nIndividual Run Results:")
                for i, (r, t) in enumerate(zip(recall_list, avg_time_list), 1):
                    logger.info(f"  Run {i}: Recall@10={r:.4f}, Time={t:.2f}ms")
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error(f"TEST FAILED: {e}")
            logger.error("="*70)
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MLDR数据集测试运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # OceanBase BM25 全文检索测试
  python mldr_test_runner.py --lang en --backend oceanbase --query-type bm25

  # OceanBase 混合检索测试
  python mldr_test_runner.py --lang zh --backend oceanbase --query-type hybrid_dense_bm25

  # 跳过数据插入（数据已存在）
  python mldr_test_runner.py --lang en --query-type bm25 --skip-insert

  # 多次测试取平均值
  python mldr_test_runner.py --lang en --query-type bm25 --runs 3 --warmup
        """
    )

    parser.add_argument('--lang', type=str, default='en',
                       help='测试语言 (默认: en)')
    parser.add_argument('--backend', type=str, default='oceanbase',
                        choices=['oceanbase'],
                       help='后端数据库 (默认: oceanbase)')
    parser.add_argument('--query-type', type=str, default='bm25',
                       choices=list(MLDRTestRunner.QUERY_TYPE_MAP.keys()),
                       help='查询类型 (默认: bm25)')
    parser.add_argument('--skip-insert', action='store_true',
                       help='跳过数据插入步骤')
    parser.add_argument('--runs', type=int, default=1,
                       help='测试运行次数，取平均值 (默认: 1)')
    parser.add_argument('--warmup', action='store_true',
                       help='执行预热查询（不计入结果）')
    parser.add_argument('--result-dir', type=str, default=None,
                       help='结果保存目录 (默认: 临时目录)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (YAML 格式)，未指定则使用默认配置')
    
    args = parser.parse_args()

    runner = MLDRTestRunner(
        lang=args.lang,
        backend=args.backend,
        query_type=args.query_type,
        skip_insert=args.skip_insert,
        result_dir=args.result_dir,
        config_file=args.config
    )

    try:
        runner.run(warmup=args.warmup, num_runs=args.runs)
        return 0
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
