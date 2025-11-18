# Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# 导入统一的缓存配置模块
from hf_cache_config import setup_hf_cache, get_cache_dir

# 设置Hugging Face缓存
# setup_hf_cache()

from itertools import chain, combinations
import time
from dataclasses import dataclass, field
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import numpy as np
import struct
import pymysql
import json
from mldr_common_tools import QueryArgs, check_languages, check_query_types
from mldr_common_tools import FakeJScoredDoc, get_queries_and_qids, save_result
from mldr_common_tools import query_yields, apply_funcs, get_colbert_model, save_colbert_list
from transformers import HfArgumentParser


# def setup_mirror():
#     """设置HF-Mirror镜像站和缓存"""
#     setup_hf_cache()

# setup_mirror()


@dataclass
class ModelArgs:
    colbert_model: str = field(default="jina-colbert", metadata={'help': 'The name of the colbert model to use.'})
    fp16: bool = field(default=True, metadata={'help': 'Use fp16 in inference?'})


@dataclass
class FusionArgs:
    fusion_method: str = field(default="rrf", metadata={'help': 'Fusion method: rrf, weighted, or normalized'})
    weights: str = field(default=None, metadata={'help': 'Weights for weighted fusion (comma-separated, e.g., "0.6,0.4")'})


def get_bge_m3_model(model_args: ModelArgs):
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=model_args.fp16)


def prepare_dense_embedding(embedding_file: str, model_args: ModelArgs, queries: list[str], qids: list[int]):
    model = get_bge_m3_model(model_args)
    query_embedding = model.encode(queries, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    query_embedding = query_embedding['dense_vecs'].astype(np.float32)
    assert len(query_embedding.shape) == 2
    assert query_embedding.shape[0] == len(qids)
    assert query_embedding.shape[1] == 1024
    with open(embedding_file, 'wb') as f:
        for single_embedding in query_embedding:
            assert len(single_embedding) == 1024
            f.write(struct.pack('i', 1024))
            single_embedding.tofile(f)
    return


def prepare_sparse_embedding(embedding_file: str, model_args: ModelArgs, queries: list[str], qids: list[int]):
    model = get_bge_m3_model(model_args)
    query_embedding = model.encode(queries, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    query_embedding = query_embedding['lexical_weights']
    assert len(query_embedding) == len(qids)
    with open(embedding_file, 'wb') as f:
        f.write(struct.pack('i', len(query_embedding)))
        for one_dict in query_embedding:
            tmp_list = []
            for p, v in one_dict.items():
                tmp_list.append((int(p), float(v)))
            tmp_list.sort()
            f.write(struct.pack('i', len(tmp_list)))
            for p, v in tmp_list:
                f.write(struct.pack('if', p, v))
    return


def prepare_colbert_embedding(embedding_file: str, model_args: ModelArgs, queries: list[str], qids: list[int]):
    model = get_colbert_model(model_args)
    query_embedding = model.encode_query(queries)
    query_embedding = query_embedding['colbert_vecs']
    assert len(query_embedding) == len(qids)
    save_colbert_list(query_embedding, embedding_file)


def powerset_above_2(s: list):
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s) + 1))


single_query_func_params = {'colbert': ('_score', 'SCORE'), 'bm25': ('_score', 'SCORE'),
                            'dense': ('_similarity', 'SIMILARITY'), 'sparse': ('_similarity', 'SIMILARITY')}


class OceanBaseClientForSearch:
    def __init__(self, with_colbert: bool):
        # OceanBase连接参数
        self.host = '127.0.0.1'
        self.port = 2881
        self.user = 'root'
        self.password = ''
        self.database = 'test'
        
        self.test_db_name = "test"
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
                charset='utf8mb4',
                autocommit=True,
                connect_timeout=60,  # 连接超时时间（秒）
                read_timeout=300,    # 读取超时时间（秒），查询可能需要较长时间
                write_timeout=300    # 写入超时时间（秒），查询可能需要较长时间
            )
            print(f"成功连接到OceanBase数据库: {self.host}:{self.port}")
            # 设置系统配置项
            self._enable_index_merge()
        except Exception as e:
            print(f"连接OceanBase数据库失败: {e}")
            raise

    def _enable_index_merge(self):
        """启用索引合并功能"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("ALTER SYSTEM SET _enable_index_merge=true;")
                print("成功启用索引合并功能: _enable_index_merge=true")
        except Exception as e:
            print(f"启用索引合并功能失败: {e}")
            # 不抛出异常，继续执行

    def get_test_table(self, language_suffix: str):
        """获取测试表"""
        table_name = self.test_table_name_prefix + language_suffix
        # table_name = self.test_table_name_prefix
        print(f"使用表: {table_name}")
        return table_name

    def common_single_query_func(self, query_type: str, query_target, max_hits: int):
        """执行单一查询类型"""
        table_name = self.test_table_name_prefix + query_target.get('language', 'en')
        
        if query_type == 'bm25':
            return self._bm25_query(table_name, query_target, max_hits)
        elif query_type == 'dense':
            return self._dense_query(table_name, query_target, max_hits)
        elif query_type == 'sparse':
            return self._sparse_query(table_name, query_target, max_hits)
        elif query_type == 'colbert':
            return self._colbert_query(table_name, query_target, max_hits)
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")

    def _bm25_query(self, table_name: str, query_target, max_hits: int):
        """BM25全文搜索查询"""
        query_text = query_target.get('query', '')
        
        # 分析查询文本的分词结果
        # self._analyze_query_tokenization(query_text, table_name)
        
        sql = f"""
        SELECT docid_col, MATCH(fulltext_col) AGAINST(%s) as _score
        FROM {table_name}
        WHERE MATCH(fulltext_col) AGAINST(%s) and base_id in ('base_id_1', 'base_id_2', 'base_id_3') and id < 1000 
        ORDER BY _score DESC
        LIMIT %s
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (query_text, query_text, max_hits))
                
                results = cursor.fetchall()
                
                docid_list = [row[0] for row in results]
                score_list = [float(row[1]) for row in results]
                
                return docid_list, score_list
        except Exception as e:
            print(f"BM25查询失败: {e}")
            return [], []

    def _analyze_query_tokenization(self, query_text, table_name):
        """分析查询文本的分词结果"""
        try:
            # 尝试获取OceanBase的分词结果
            # 方法1：使用EXPLAIN查看查询计划
            explain_sql = f"SELECT tokenize(%s,'space') from test_insert_en limit 1;"
            
            with self.connection.cursor() as cursor:
                cursor.execute(explain_sql, (query_text,))
                explain_result = cursor.fetchall()
                
                # 保存分词分析结果到文件
                
                with open('word.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\n=== 查询分词分析 ===\n")
                    f.write(f"查询文本: {query_text}\n")
                    f.write(f"表名: {table_name}\n")
                    f.write(f"EXPLAIN结果:\n")
                    for row in explain_result:
                        f.write(f"  {row}\n")
                    f.write("-" * 50 + "\n")
                
                print(f"查询分词分析已保存到 word.txt")
                
        except Exception as e:
            print(f"分词分析失败: {e}")
            
            # 如果无法获取OceanBase的分词结果，使用简单的分词分析
            simple_tokens = self._simple_tokenize(query_text)
            
            with open('word.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n=== 简单分词分析 ===\n")
                f.write(f"查询文本: {query_text}\n")
                f.write(f"简单分词结果: {simple_tokens}\n")
                f.write(f"分词数量: {len(simple_tokens)}\n")
                f.write("-" * 50 + "\n")
            
            print(f"简单分词结果: {simple_tokens}")

    def _simple_tokenize(self, text):
        """简单的英文分词分析"""
        import re
        
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号（保留下划线和连字符）
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # 分割成词条
        tokens = text.split()
        
        # 过滤空字符串
        tokens = [token for token in tokens if token.strip()]
        
        return tokens

    def _dense_query(self, table_name: str, query_target, max_hits: int):
        """密集向量相似度查询，带filter和where字段，先改为全文字段场景"""
        query_vector = query_target.get('dense_vector', [])
        if not query_vector:
            return [], []
         
        vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        
        sql = f"""
        SELECT docid_col
        FROM {table_name}
        WHERE MATCH(fulltext_col) AGAINST(%s) and base_id in ('base_id_1', 'base_id_2', 'base_id_3')
        LIMIT %s
        """ 
        
        try:
            with self.connection.cursor() as cursor:
                print(f"调试: 执行SQL: {sql}")
                print(f"调试: 参数: {vector_str}")
                print(f"调试: 最大结果数: {max_hits}")
                cursor.execute(sql, (vector_str, max_hits))
                results = cursor.fetchall()
                
                docid_list = [row[0] for row in results]
                score_list = [float(row[1]) for row in results]
                
                return docid_list, score_list
        except Exception as e:
            print(f"密集向量查询失败: {e}")
            return [], []

    def _sparse_query(self, table_name: str, query_target, max_hits: int):
        """稀疏向量相似度查询"""
        sparse_data = query_target.get('sparse_vector', {})
        if not sparse_data or 'indices' not in sparse_data or 'values' not in sparse_data:
            return [], []
        
        # 将稀疏向量转换为OceanBase SPARSEVECTOR格式
        sparse_pairs = []
        for idx, val in zip(sparse_data['indices'], sparse_data['values']):
            sparse_pairs.append(f"{idx}:{val}")
        sparse_vector_str = "{" + ",".join(sparse_pairs) + "}"
        
        sql = f"""
        SELECT docid_col, INNER_PRODUCT(sparse_col, %s) as _similarity
        FROM {table_name}
        WHERE sparse_col IS NOT NULL
            ORDER BY _similarity DESC
            LIMIT %s
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (sparse_vector_str, max_hits))
                results = cursor.fetchall()
                
                docid_list = [row[0] for row in results]
                score_list = [float(row[1]) for row in results]
                
                return docid_list, score_list
        except Exception as e:
            print(f"稀疏向量查询失败: {e}")
            return [], []

    def _colbert_query(self, table_name: str, query_target, max_hits: int):
        """ColBERT查询（如果支持）"""
        if not self.with_colbert:
            print("警告: ColBERT查询被请求但未启用")
            return [], []
        
        # ColBERT查询实现（需要根据OceanBase的ColBERT支持情况调整）
        query_text = query_target.get('query', '')
        
        sql = f"""
        SELECT docid_col, 1.0 as _score
        FROM {table_name}
        WHERE fulltext_col LIKE %s
        LIMIT %s
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (f"%{query_text}%", max_hits))
                results = cursor.fetchall()
                
                docid_list = [row[0] for row in results]
                score_list = [float(row[1]) for row in results]
                
                return docid_list, score_list
        except Exception as e:
            print(f"ColBERT查询失败: {e}")
            return [], []

    def fusion_query(self, query_targets_list: list, apply_funcs_list: list, max_hits: int):
        """融合查询 - 使用RRF (Reciprocal Rank Fusion)"""
        # 收集所有查询结果
        all_results = {}
        
        print(f"调试: 开始RRF融合，查询类型数量: {len(query_targets_list)}")
        
        for i, (query_target, apply_func) in enumerate(zip(query_targets_list, apply_funcs_list)):
            query_type = query_target.get('type', 'bm25')
            print(f"调试: 执行查询 {i+1}, 类型: {query_type}")
            
            # 执行单个查询
            docid_list, score_list = self.common_single_query_func(
                query_type, 
                query_target, 
                max_hits
            )
            
            print(f"调试: 查询 {i+1} 返回 {len(docid_list)} 个结果")
            if len(docid_list) > 0:
                print(f"调试: 查询 {i+1} 前3个结果: {docid_list[:3]}")
                print(f"调试: 查询 {i+1} 前3个分数: {score_list[:3]}")
            
            # 根据查询类型调整分数方向
            if query_type == 'dense':
                # 将L2距离转换为相似度分数（取负值）
                score_list = [-score for score in score_list]
                print(f"调试: 查询 {i+1} (dense) 分数转换后前3个: {score_list[:3]}")
            
            # 计算RRF分数
            for rank, (docid, score) in enumerate(zip(docid_list, score_list)):
                if docid not in all_results:
                    all_results[docid] = 0.0
                # RRF公式: 1 / (k + rank)
                rrf_score = 1.0 / (60.0 + rank + 1)  # k=60
                all_results[docid] += rrf_score
        
        print(f"调试: RRF融合完成，总共 {len(all_results)} 个唯一文档")
        
        # 按RRF分数排序
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前max_hits个结果
        docid_list = [item[0] for item in sorted_results[:max_hits]]
        score_list = [item[1] for item in sorted_results[:max_hits]]
        
        print(f"调试: 最终返回 {len(docid_list)} 个融合结果")
        if len(docid_list) > 0:
            print(f"调试: 融合结果前3个: {docid_list[:3]}")
            print(f"调试: 融合分数前3个: {score_list[:3]}")
        
        return docid_list, score_list

    def fusion_query_with_scores(self, query_targets_list: list, apply_funcs_list: list, max_hits: int, 
                                fusion_method='rrf', weights=None):
        """改进的融合查询 - 支持多种融合方法"""
        if fusion_method == 'rrf':
            return self.fusion_query(query_targets_list, apply_funcs_list, max_hits)
        
        # 收集所有查询结果
        all_results = {}
        query_results = {}
        
        for i, (query_target, apply_func) in enumerate(zip(query_targets_list, apply_funcs_list)):
            # 执行单个查询
            docid_list, score_list = self.common_single_query_func(
                query_target.get('type', 'bm25'), 
                query_target, 
                max_hits
            )
            
            # 根据查询类型调整分数方向
            query_type = query_target.get('type', 'bm25')
            if query_type == 'dense':
                # 将L2距离转换为相似度分数（取负值）
                score_list = [-score for score in score_list]
            
            # 存储每个查询的结果
            query_results[i] = {docid: score for docid, score in zip(docid_list, score_list)}
            
            # 归一化分数到[0,1]范围
            if score_list:
                min_score, max_score = min(score_list), max(score_list)
                if max_score > min_score:
                    normalized_scores = [(score - min_score) / (max_score - min_score) for score in score_list]
                else:
                    normalized_scores = [1.0] * len(score_list)
                
                for docid, norm_score in zip(docid_list, normalized_scores):
                    if docid not in all_results:
                        all_results[docid] = 0.0
                    
                    # 应用权重
                    weight = weights[i] if weights and i < len(weights) else 1.0
                    all_results[docid] += weight * norm_score
        
        # 按融合分数排序
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前max_hits个结果
        docid_list = [item[0] for item in sorted_results[:max_hits]]
        score_list = [item[1] for item in sorted_results[:max_hits]]
        
        return docid_list, score_list

    def main(self, languages: list[str], query_types: list[str], model_args: ModelArgs, save_dir: str):
        """主函数"""
        for lang in languages:
            print(f"开始搜索语言: {lang}")
            table_name = self.get_test_table(lang)
            
            # 加载查询数据
            # 注意：即使使用streaming=True，get_queries_and_qids也会将所有数据加载到内存
            # 所以这里直接加载，但在使用完后会显式释放
            queries, qids = get_queries_and_qids(lang, False)
            print(f"查询总数: {len(qids)}")
            if len(queries) > 0:
                print(f"平均查询字符数: {sum([len(q) for q in queries]) / len(queries):.1f}")
            
            for query_type in query_types:
                embedding_file = None
                if query_type in self.prepare_embedding_funcs:
                    embedding_file = os.path.join(save_dir, f"query_embedding_{lang}_{query_type}.data")
                    if not os.path.exists(embedding_file):
                        print(f"开始准备查询嵌入: {query_type}")
                        self.prepare_embedding_funcs[query_type](embedding_file, model_args, queries, qids)
                
                query_yield = query_yields[query_type](queries, embedding_file)
                print(f"开始搜索查询方法: {query_type}")
                
                total_query_time: float = 0.0
                total_query_num: int = len(qids)
                result_list = []
                
                for query, qid in tqdm(zip(queries, qids), total=total_query_num):
                    query_target = next(query_yield)
                    
                    # 根据查询类型处理query_target
                    if query_type == 'dense':
                        # 密集向量查询：query_target是向量列表
                        query_target_dict = {'dense_vector': query_target, 'language': lang}
                    elif query_type == 'bm25':
                        # BM25查询：query_target是查询字符串
                        query_target_dict = {'query': query_target, 'language': lang}
                    else:
                        # 其他类型：query_target是字典
                        query_target_dict = query_target
                        query_target_dict['language'] = lang
                    
                    time_start = time.time()
                    docid_list, score_list = self.common_single_query_func(query_type, query_target_dict, max_hits=10)
                    time_end = time.time()
                    total_query_time += time_end - time_start
                    
                    result = []
                    for docid, score in zip(docid_list, score_list):
                        result.append(FakeJScoredDoc(docid, score))
                    result_list.append((qid, result))
                
                print(f"平均查询时间: {1000.0 * total_query_time / float(total_query_num)} ms.")
                save_path = os.path.join(save_dir, f"{lang}_{query_type}.txt")
                print(f"保存搜索结果到: {save_path}")
                save_result(result_list, save_path, qids, max_hits=10)
                
                # 释放结果列表内存
                del result_list
                import gc
                gc.collect()
            
            # 释放查询数据内存（如果不再需要）
            if len(query_types) < 2:
                del queries, qids
                import gc
                gc.collect()
                return
            
            # 融合操作
            print("开始搜索融合方法。")
            for query_type_comb in powerset_above_2(query_types):
                query_type_comb_str = '_'.join(query_type_comb)
                print(f"开始搜索融合方法: {query_type_comb_str}")
                
                embedding_files_list = [os.path.join(save_dir, f"query_embedding_{lang}_{query_type}.data") for
                                        query_type in query_type_comb]
                query_yields_list = [query_yields[query_type](queries, embedding_file) for query_type, embedding_file in
                                     zip(query_type_comb, embedding_files_list)]
                apply_funcs_list = [apply_funcs[query_type] for query_type in query_type_comb]
                
                total_query_time: float = 0.0
                total_query_num: int = len(qids)
                result_list = []
                
                for query, qid in tqdm(zip(queries, qids), total=total_query_num):
                    query_targets_list = []
                    for query_yield, query_type in zip(query_yields_list, query_type_comb):
                        query_target = next(query_yield)
                        
                        # 根据查询类型处理query_target
                        if query_type == 'dense':
                            # 密集向量查询：query_target是向量列表
                            query_target_dict = {'dense_vector': query_target, 'language': lang, 'type': query_type}
                        elif query_type == 'bm25':
                            # BM25查询：query_target是查询字符串
                            query_target_dict = {'query': query_target, 'language': lang, 'type': query_type}
                        else:
                            # 其他类型：query_target是字典
                            query_target_dict = query_target
                            query_target_dict['language'] = lang
                            query_target_dict['type'] = query_type
                        
                        query_targets_list.append(query_target_dict)
                    
                    time_start = time.time()
                    # 根据融合方法选择不同的融合策略
                    if hasattr(self, 'fusion_method') and self.fusion_method == 'weighted':
                        # 解析权重
                        weights = None
                        if hasattr(self, 'weights') and self.weights:
                            weights = [float(w.strip()) for w in self.weights.split(',')]
                        docid_list, score_list = self.fusion_query_with_scores(
                            query_targets_list, apply_funcs_list, max_hits=10, 
                            fusion_method='weighted', weights=weights
                        )
                    else:
                        # 默认使用RRF
                        docid_list, score_list = self.fusion_query(query_targets_list, apply_funcs_list, max_hits=10)
                    time_end = time.time()
                    total_query_time += time_end - time_start
                    
                    result = []
                    for docid, score in zip(docid_list, score_list):
                        result.append(FakeJScoredDoc(docid, score))
                    result_list.append((qid, result))
                
                print(f"平均查询时间: {1000.0 * total_query_time / float(total_query_num)} ms.")
                save_path = os.path.join(save_dir, f"{lang}_fusion_{query_type_comb_str}.txt")
                print(f"保存搜索结果到: {save_path}")
                save_result(result_list, save_path, qids, max_hits=10)
                
                # 释放结果列表内存
                del result_list
                import gc
                gc.collect()
            
            # 融合查询完成后，释放查询数据内存
            if len(query_types) >= 2:
                del queries, qids
                import gc
                gc.collect()

    @property
    def prepare_embedding_funcs(self):
        """准备嵌入函数映射"""
        return {'dense': prepare_dense_embedding, 
                # 'sparse': prepare_sparse_embedding,
                'colbert': prepare_colbert_embedding}


if __name__ == "__main__":
    parser = HfArgumentParser([ModelArgs, QueryArgs, FusionArgs])
    model_args, query_args, fusion_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    query_args: QueryArgs
    fusion_args: FusionArgs
    
    languages = check_languages(query_args.languages)
    query_types = check_query_types(query_args.query_types)
    
    if 'colbert' in query_types and not query_args.with_colbert:
        raise ValueError("启用了Colbert查询类型但with_colbert为False。")
    
    oceanbase_client = OceanBaseClientForSearch(query_args.with_colbert)
    
    oceanbase_client.fusion_method = fusion_args.fusion_method
    oceanbase_client.weights = fusion_args.weights
    
    try:
        oceanbase_client.main(languages=languages, query_types=query_types, model_args=model_args,
                             save_dir=query_args.query_result_dave_dir)
    finally:
        if oceanbase_client.connection:
            oceanbase_client.connection.close()
            print("数据库连接已关闭") 