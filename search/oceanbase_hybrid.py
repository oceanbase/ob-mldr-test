#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanBase 混合搜索模块
支持：原生 Hybrid Search 语法（Dense+BM25）
"""

import os
import sys
import time
import struct
import pymysql
import json
import argparse
import numpy as np
from tqdm import tqdm
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import (
    check_languages,
    check_query_types,
    get_default_query_result_dir,
    FakeJScoredDoc,
    get_queries_and_qids,
    save_result,
    query_yields,
    # 向量生成
    ModelArgs,
    prepare_dense_embedding,
    # prepare_colbert_embedding
)

# 导入配置
from config import OceanBaseHybridConfig
from config_loader import load_oceanbase_hybrid_config, load_embedding_config


class OceanBaseClientForHybridSearch:
    def __init__(self, with_colbert: bool, config: OceanBaseHybridConfig = None):
        """
        初始化OceanBase混合搜索客户端

        Args:
            with_colbert: 是否支持ColBERT
            config: 数据库配置，None则使用默认配置
        """
        if config is None:
            config = OceanBaseHybridConfig()

        self.host = config.host
        self.port = config.port
        self.user = config.user
        self.password = config.password
        self.database = config.database
        self.charset = config.charset
        self.autocommit = config.autocommit

        self.test_db_name = config.database
        self.test_table_name_prefix = "test_insert_"
        self.connection = None
        self.with_colbert = with_colbert
        self.connect()

    def connect(self):
        """连接到OceanBase数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=self.autocommit
            )
        except Exception as e:
            logging.error(f"连接OceanBase数据库失败: {e}")
            raise

    def get_test_table(self, language_suffix: str):
        """获取测试表"""
        return self.test_table_name_prefix + language_suffix

    def common_single_query_func(self, query_type: str, query_target, max_hits: int):
        """执行单一查询类型，返回 (docid_list, score_list, execution_time)"""
        table_name = self.test_table_name_prefix + query_target.get('language', 'en')

        if query_type == 'bm25':
            return self._bm25_hybrid_query(table_name, query_target, max_hits)
        elif query_type == 'dense':
            return self._dense_hybrid_query(table_name, query_target, max_hits)
        elif query_type == 'colbert':
            return self._colbert_hybrid_query(table_name, query_target, max_hits)
        elif query_type == 'hybrid_dense_bm25':
            return self._dense_bm25_hybrid_query(table_name, query_target, max_hits)
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")

    def _bm25_hybrid_query(self, table_name: str, query_target, max_hits: int):
        """BM25全文搜索查询"""
        query_text = query_target.get('query', '')
        query_text = query_text.replace('\\', '\\\\')
        query_text = query_text.replace('"', '\\"')
        query_text = query_text.replace("'", "\\'")
        query_text = query_text.replace('\n', '\\n')
        query_text = query_text.replace('\r', '\\r')
        query_text = query_text.replace('\t', '\\t')
        hybrid_params = {
            "query": {
                "query_string": {
                    "fields": ["fulltext_col"],
                    "query": query_text,
                    "boost": 1.0
                }
            },
            "size": max_hits,
            "_source": ["docid_col"]
        }

        docid_list, score_list, exec_time = self._execute_hybrid_search(table_name, hybrid_params, max_hits)
        return docid_list, score_list, exec_time

    def _dense_hybrid_query(self, table_name: str, query_target, max_hits: int):
        """密集向量相似度查询"""
        query_vector = query_target.get('dense_vector', [])
        if not query_vector:
            return [], [], 0.0

        vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        knn_params = {
            "knn": {
                "field": "dense_col",
                "k": max_hits,
                "query_vector": vector_str
            },
            "size": max_hits,
            "_source": ["docid_col"]
        }

        docid_list, score_list, exec_time = self._execute_hybrid_search(table_name, knn_params, max_hits)
        return docid_list, score_list, exec_time

    def _colbert_hybrid_query(self, table_name: str, query_target, max_hits: int):
        """ColBERT查询 - hybrid search格式"""
        if not self.with_colbert:
            return [], [], 0.0

        query_text = query_target.get('query', '')

        # 构建hybrid search参数
        hybrid_params = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"fulltext_col": query_text}}
                    ]
                }
            },
            "knn": {
                "field": "dense_col",
                "k": max_hits,
                "query_vector": "[0.0]" * 1024  # 占位向量
            },
            "_source": ["docid_col", "_keyword_score", "_semantic_score"]
        }

        docid_list, score_list, exec_time = self._execute_hybrid_search(table_name, hybrid_params, max_hits)
        return docid_list, score_list, exec_time

    def _dense_bm25_hybrid_query(self, table_name: str, query_target, max_hits: int):
        """密集向量和BM25结合的混合查询"""
        query_text = query_target.get('query', '')
        query_vector = query_target.get('dense_vector', [])

        if not query_vector:
            return self._bm25_hybrid_query(table_name, query_target, max_hits)

        query_text = query_text.replace('\\', '\\\\')
        query_text = query_text.replace('"', '\\"')
        query_text = query_text.replace("'", "\\'")
        query_text = query_text.replace('\n', '\\n')
        query_text = query_text.replace('\r', '\\r')
        query_text = query_text.replace('\t', '\\t')

        vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        # 使用全文搜索和向量搜索
        hybrid_params = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "fields": ["fulltext_col"],
                                "query": query_text,
                                "boost": 1.0
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": "dense_col",
                "k": max_hits,
                "query_vector": vector_str
            },
            "size": max_hits,
            "_source": ["docid_col"]
        }

        docid_list, score_list, exec_time = self._execute_hybrid_search(table_name, hybrid_params, max_hits)
        return docid_list, score_list, exec_time

    def _execute_hybrid_search(self, table_name: str, hybrid_params: dict, max_hits: int):
        """执行hybrid search查询"""
        try:
            params_json = json.dumps(hybrid_params, ensure_ascii=False)

            with self.connection.cursor() as cursor:
                # 执行原子性hybrid search查询
                atomic_sql = "SELECT dbms_hybrid_search.search(%s, %s)"
                # 记录SQL实际执行时间
                sql_start_time = time.time()
                cursor.execute(atomic_sql, (table_name, params_json))
                result = cursor.fetchone()
                sql_end_time = time.time()
                sql_execution_time = sql_end_time - sql_start_time

                if result and result[0]:
                    try:
                        result_json = json.loads(result[0])

                        docid_list = []
                        score_list = []

                        # 支持两种JSON格式：数组格式和标准格式
                        if isinstance(result_json, list):
                            # 数组格式：直接包含结果列表
                            for hit in result_json:
                                if isinstance(hit, dict):
                                    docid = hit.get('docid_col', '')
                                    if docid:
                                        docid_list.append(docid)

                                        # 优先使用特定的score字段
                                        score = hit.get('_score', 0.0)
                                        if '_keyword_score' in hit and hit['_keyword_score'] is not None:
                                            score = hit['_keyword_score']
                                        elif '_semantic_score' in hit and hit['_semantic_score'] is not None:
                                            score = hit['_semantic_score']
                                        score_list.append(float(score) if score is not None else 0.0)

                        elif isinstance(result_json, dict) and 'hits' in result_json and 'hits' in result_json['hits']:
                            # 标准格式：包含hits嵌套结构
                            hits = result_json['hits']['hits']

                            for hit in hits:
                                if '_source' in hit and 'docid_col' in hit['_source']:
                                    docid = hit['_source']['docid_col']
                                    docid_list.append(docid)

                                    # 获取分数
                                    score = hit.get('_score', 0.0)
                                    if '_source' in hit:
                                        if '_keyword_score' in hit['_source'] and hit['_source']['_keyword_score'] is not None:
                                            score = hit['_source']['_keyword_score']
                                        elif '_semantic_score' in hit['_source'] and hit['_source']['_semantic_score'] is not None:
                                            score = hit['_source']['_semantic_score']
                                    
                                    score_list.append(float(score) if score is not None else 0.0)
                        
                        return docid_list, score_list, sql_execution_time
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"警告: 无法解析hybrid search结果JSON: {e}")
                        return [], [], 0.0
                    except Exception as e:
                        logging.warning(f"警告: 解析结果时发生错误: {e}")
                        import traceback
                        traceback.print_exc()
                        return [], [], 0.0
                else:
                    return [], [], 0.0

        except Exception as e:
            logging.error(f"Hybrid search查询失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()

    def main(self, languages: list[str], query_types: list[str], model_args: ModelArgs, save_dir: str) -> dict:
        """主函数，返回每种查询类型的平均SQL执行时间（毫秒）"""
        avg_times = {}

        for lang in languages:
            logging.info(f"开始搜索语言: {lang}")
            table_name = self.get_test_table(lang)

            queries, qids = get_queries_and_qids(lang, False)
            logging.info(f"查询总数: {len(qids)}")

            for query_type in query_types:
                embedding_file = None
                if query_type in self.prepare_embedding_funcs:
                    embedding_file = os.path.join(save_dir, f"query_embedding_{lang}_{query_type}.data")
                    if not os.path.exists(embedding_file):
                        self.prepare_embedding_funcs[query_type](embedding_file, model_args, queries, qids)
                
                total_query_time: float = 0.0
                total_query_num: int = len(qids)
                result_list = []

                if query_type == 'hybrid_dense_bm25':
                    embedding_file = os.path.join(save_dir, f"query_embedding_{lang}_dense.data")
                    if not os.path.exists(embedding_file):
                        prepare_dense_embedding(embedding_file, model_args, queries, qids)

                    bm25_query_yield = query_yields['bm25'](queries, None)
                    
                    dense_vectors = []
                    with open(embedding_file, 'rb') as f:
                        for _ in range(len(queries)):
                            vector_size = struct.unpack('i', f.read(4))[0]
                            vector = np.frombuffer(f.read(vector_size * 4), dtype=np.float32)
                            dense_vectors.append(vector.tolist())
                    
                    for query, qid, processed_query, dense_vector in tqdm(zip(queries, qids, bm25_query_yield, dense_vectors), total=total_query_num):
                        query_target_dict = {
                            'query': processed_query,
                            'dense_vector': dense_vector,
                            'language': lang
                        }
                        
                        docid_list, score_list, sql_exec_time = self._dense_bm25_hybrid_query(
                            self.test_table_name_prefix + lang, 
                            query_target_dict, 
                            max_hits=1000
                        )
                        total_query_time += sql_exec_time
                        
                        result = []
                        for docid, score in zip(docid_list, score_list):
                            result.append(FakeJScoredDoc(docid, score))
                        result_list.append((qid, result))
                else:
                    query_yield = query_yields[query_type](queries, embedding_file)
                    
                    for query, qid in tqdm(zip(queries, qids), total=total_query_num):
                        query_target = next(query_yield)
                        
                        if query_type == 'dense':
                            query_target_dict = {'dense_vector': query_target, 'language': lang}
                        elif query_type == 'bm25':
                            query_target_dict = {'query': query_target, 'language': lang}
                        else:
                            query_target_dict = query_target
                            query_target_dict['language'] = lang
                        
                        docid_list, score_list, sql_exec_time = self.common_single_query_func(query_type, query_target_dict, max_hits=1000)
                        total_query_time += sql_exec_time
                        
                        result = []
                        for docid, score in zip(docid_list, score_list):
                            result.append(FakeJScoredDoc(docid, score))
                        result_list.append((qid, result))
                
                avg_query_time_ms = 1000.0 * total_query_time / float(total_query_num)
                logging.info(f"平均SQL执行时间: {avg_query_time_ms} ms.")

                key = f"{lang}_{query_type}"
                avg_times[key] = avg_query_time_ms

                save_path = os.path.join(save_dir, f"{lang}_{query_type}.txt")
                save_result(result_list, save_path, qids, max_hits=1000)
        
        return avg_times

    @property
    def prepare_embedding_funcs(self):
        """准备嵌入函数映射"""
        return {
            'dense': prepare_dense_embedding, 
            'hybrid_dense_bm25': prepare_dense_embedding,
            # 'colbert': prepare_colbert_embedding
        }


def main():
    """主函数入口"""
    parser = argparse.ArgumentParser(description='OceanBase混合搜索脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（YAML 格式），未指定则使用默认配置')
    
    # 语言参数
    parser.add_argument('--languages', nargs='+', default=['en'],
                       help='要处理的语言列表，支持: ar de en es fr hi it ja ko pt ru th zh')
    parser.add_argument('--all', action='store_true',
                       help='处理所有支持的语言')
    
    # 查询类型参数
    parser.add_argument('--query-types', nargs='+', 
                       default=['bm25', 'dense', 'hybrid_dense_bm25'],
                       help='查询类型列表，支持: bm25, dense, colbert, hybrid_dense_bm25')
    
    # ColBERT支持
    parser.add_argument('--with-colbert', action='store_true',
                       help='启用ColBERT支持')
    parser.add_argument('--model-name-or-path', type=str, default=None,
                       help='模型路径或名称（覆盖配置文件）')
    parser.add_argument('--pooling-method', type=str, default=None,
                       help='池化方法（覆盖配置文件）')
    parser.add_argument('--normalize-embeddings', type=bool, default=None,
                       help='是否归一化嵌入（覆盖配置文件）')
    parser.add_argument('--use-fp16', type=bool, default=None,
                       help='是否使用FP16精度（覆盖配置文件）')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='结果保存目录，未指定则自动生成')

    args = parser.parse_args()

    config = load_oceanbase_hybrid_config(args.config)
    embedding_config = load_embedding_config(args.config)

    if args.all:
        languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    else:
        languages = check_languages(args.languages)

    query_types = check_query_types(args.query_types)

    if 'colbert' in query_types and not args.with_colbert:
        raise ValueError("启用了Colbert查询类型但with_colbert为False。")

    if not args.save_dir:
        save_dir = get_default_query_result_dir(languages)
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    model_args = ModelArgs(
        model_name_or_path=args.model_name_or_path or embedding_config.model_name_or_path,
        colbert_model=embedding_config.colbert_model,
        pooling_method=args.pooling_method or embedding_config.pooling_method,
        normalize_embeddings=args.normalize_embeddings or embedding_config.normalize_embeddings,
        use_fp16=args.use_fp16 or embedding_config.use_fp16,
        max_length=embedding_config.max_length,
        batch_size=embedding_config.batch_size
    )

    oceanbase_client = OceanBaseClientForHybridSearch(
        with_colbert=args.with_colbert,
        config=config
    )

    try:
        oceanbase_client.main(
            languages=languages,
            query_types=query_types,
            model_args=model_args,
            save_dir=save_dir
        )
    except Exception as e:
        logging.error(f"✗ 混合搜索测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        oceanbase_client.close()


if __name__ == "__main__":
    main()
 