#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanBase 数据插入模块
功能：插入测试数据并创建索引
"""

import os
import time
import sys
import pymysql
import argparse
from typing import Optional
from tqdm import tqdm
import logging

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import (
    load_corpus,
    check_languages,
    ModelArgs,
    generate_and_save_dense_vectors,
    download_dense_vectors,
    VectorFileIterator,
    generate_vector_filename,
    generate_dense_vectors_from_stream,
    load_dense_vectors_from_file
)
from config import OceanBaseInsertConfig
from config_loader import load_oceanbase_insert_config, load_embedding_config


# ============================================
# 主插入函数
# ============================================
def test_simple_insert(lang: str, config: Optional[OceanBaseInsertConfig] = None):
    """
    插入数据测试
    
    Args:
        lang: 语言代码
        config: 配置对象，None则使用默认配置
    """
    # 使用默认配置
    if config is None:
        config = OceanBaseInsertConfig()

    filepath = ""
    embedding_config = load_embedding_config()
    corpus = load_corpus(lang, streaming=config.enable_streaming)

    if embedding_config.download_vectors:
        filepath = download_dense_vectors(lang=lang)
    else:
        model_name = embedding_config.model_name_or_path.split('/')[-1]
        vector_file = os.path.join('vectors', generate_vector_filename(lang, config.test_count, model_name))
        model_args = ModelArgs(fp16=config.use_fp16)
        if config.enable_streaming:
            filepath = generate_dense_vectors_from_stream(
                text_stream=corpus, 
                model_args=model_args, output_file=vector_file, 
                batch_size=config.vector_batch_size, total=config.test_count
            )
        else:
            filepath = generate_and_save_dense_vectors(
                    texts=corpus["text"],
                    model_args=model_args, output_file=vector_file, 
                    batch_size=config.vector_batch_size)

    conn = pymysql.connect(**config.to_db_config())
    cursor = conn.cursor()

    table_name = f"test_insert_{lang}"
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_sql)
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        id BIGINT AUTO_INCREMENT,
        base_id VARCHAR(255) NOT NULL,
        docid_col VARCHAR(255) NOT NULL,
        fulltext_col LONGTEXT,
        dense_col VECTOR(1024),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        primary key (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    heap_sql = f"alter system set default_table_organization = 'heap';"
    cursor.execute(heap_sql)
    time.sleep(3)
    cursor.execute(create_table_sql)

    test_count = config.test_count

    insert_sql = f"""
    INSERT INTO {table_name} (base_id, docid_col, fulltext_col, dense_col)
    VALUES (%s, %s, %s, %s)
    """

    batch_size = config.batch_size
    inserted_count = 0
    processed_count = 0
    failed_batches = []
    batch_data = []
    batch_num = 0
    
    try:
        if config.enable_streaming:
            with VectorFileIterator(filepath) as vec_iter, \
                tqdm(desc="流式插入", total=config.test_count, unit="条", ncols=80) as pbar:

                for data in corpus:
                    if processed_count >= config.test_count:
                        break

                    try:
                        docid = data['docid']
                        text = data['text']
                        
                        # 从向量迭代器获取下一个向量
                        try:
                            vector = next(vec_iter)
                        except StopIteration:
                            logging.warning("向量数量不足，提前终止")
                            break

                        # 构造 base_id
                        base_group_size = 500
                        base_id_number = (processed_count // base_group_size) + 1
                        base_id = f"base_id_{base_id_number}"
                        
                        # 转换为 OceanBase VECTOR 字符串格式
                        vector_str = '[' + ','.join(map(str, vector)) + ']'
                        batch_data.append((base_id, docid, text, vector_str))

                        # 批量提交
                        if len(batch_data) >= config.batch_size:
                            batch_num += 1
                            try:
                                cursor.executemany(insert_sql, batch_data)
                                conn.commit()
                                inserted_count += len(batch_data)
                                batch_data.clear()
                            except Exception as e:
                                logging.error(f"✗ 批量插入失败（批次 {batch_num}）: {e}")
                                import traceback
                                traceback.print_exc()
                                conn.rollback()
                                failed_batches.append((batch_num, str(e)))
                                batch_data.clear()
                                # 继续处理下一批，不中断

                        processed_count += 1
                        pbar.update(1)

                    except Exception as e:
                        logging.error(f"处理第 {processed_count + 1} 条记录失败: {e}")
                        continue
        else:
            docid_list = corpus["docid"]
            corpus_text_list = corpus["text"]
        
            # 限制数量
            actual_count = min(test_count, len(docid_list))
            docid_list = docid_list[:actual_count]
            corpus_text_list = corpus_text_list[:actual_count]
            vectors = load_dense_vectors_from_file(filepath, min(test_count, len(docid_list)))
            for i in tqdm(range(min(test_count, len(docid_list))), desc="插入进度"):
                processed_count = i + 1
                try:
                    docid = docid_list[i]
                    text = corpus_text_list[i]

                    base_group_size = 500
                    base_id_number = (i // base_group_size) + 1
                    base_id = f"base_id_{base_id_number}"

                    vector = vectors[i]
                    vector_str = '[' + ','.join(map(str, vector)) + ']'
                    batch_data.append((base_id, docid, text, vector_str))

                    if len(batch_data) >= batch_size:
                        batch_num += 1
                        try:
                            cursor.executemany(insert_sql, batch_data)
                            conn.commit()
                            inserted_count += len(batch_data)
                            batch_data.clear()
                        except Exception as e:
                            logging.error(f"✗ 批量插入失败（批次 {batch_num}）: {e}")
                            conn.rollback()
                            failed_batches.append((batch_num, str(e)))
                            batch_data.clear()
                
                except Exception as e:
                    logging.error(f"处理第 {i+1} 条记录失败: {e}")
                    continue
        
        if len(batch_data) > 0:
            batch_num += 1
            try:
                cursor.executemany(insert_sql, batch_data)
                conn.commit()
                inserted_count += len(batch_data)
                batch_data.clear()
            except Exception as e:
                logging.error(f"✗ 批量插入剩余数据失败: {e}")
                import traceback
                traceback.print_exc()
                conn.rollback()
                failed_batches.append((batch_num, str(e)))

    finally:
        if config.enable_streaming:
            del corpus
        import gc
        gc.collect()

    if failed_batches:
        logging.error(f"\n⚠️  共有 {len(failed_batches)} 个批次插入失败:")
        for batch_num, error in failed_batches:
            logging.error(f"  批次 {batch_num}: {error}")

    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    if count != inserted_count:
        logging.warning(f"\n⚠️  警告：数据库中的记录数 ({count}) 与插入计数 ({inserted_count}) 不一致！")
    logging.info("\n" + "="*60)
    logging.info("正在创建索引，这可能需要一些时间，请耐心等待...")
    logging.info("="*60)
    
    # 2. 全文索引
    logging.info("正在创建全文索引...")
    try:
        cursor.execute("SET ob_query_timeout = 3600000000")
        cursor.execute("SET ob_trx_timeout = 864000000")
        cursor.execute(f"CREATE /*+ parallel(16) */ FULLTEXT INDEX ft_fulltext ON {table_name}(fulltext_col) WITH PARSER space PARSER_PROPERTIES=(min_token_size=3,max_token_size=84)")
        logging.info("✓ 全文索引创建成功")
    except Exception as e:
        logging.warning(f"✗ 全文索引创建失败: {e}")

    # 3. 向量索引
    logging.info("正在创建向量索引...")
    try:
        index_sql = f"""
            CREATE /*+ parallel(16) */ VECTOR INDEX hnsw_index ON {table_name}(dense_col)
            WITH (distance=l2, type=hnsw, m=16, EF_CONSTRUCTION=200, LIB=VSAG, EF_SEARCH=300)
        """
        cursor.execute(index_sql)
        logging.info("✓ 向量索引创建成功")
    except Exception as e:
        logging.warning(f"✗ 向量索引创建失败: {e}")
    
    logging.info("="*60)
    logging.info("✓ 所有索引创建完成！")
    logging.info("="*60)

    logging.info("合并...")
    try:
        merge_start = time.time()
        cursor.execute(f"ALTER SYSTEM major freeze;")
        time.sleep(3)
        _wait_for_major_compaction(cursor)
        merge_end = time.time()
        merge_duration = merge_end - merge_start
        logging.info(f"合并成功，耗时: {format_duration(merge_duration)}")
    except Exception as e:
        logging.warning(f"合并失败: {e}")

    logging.info("收集统计信息...")
    try:
        analyze_start = time.time()
        cursor.execute(f"ANALYZE TABLE {table_name};")
        analyze_end = time.time()
        analyze_duration = analyze_end - analyze_start
        logging.info(f"统计信息收集成功，耗时: {format_duration(analyze_duration)}")
    except Exception as e:
        logging.warning(f"统计信息收集失败: {e}")

    cursor.close()
    conn.close()

def _wait_for_major_compaction(cursor):
    while True:
        cursor.execute(
            "SELECT IF(COUNT(*) = COUNT(STATUS = 'IDLE' OR NULL), 'TRUE', 'FALSE') "
            "AS all_status_idle FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION;"
        )
        all_status_idle = cursor.fetchone()[0]
        if all_status_idle == "TRUE":
            break
        time.sleep(30)
        logging.info("合并中")

def format_duration(seconds):
    """将秒数格式化为分钟和秒数"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes} 分 {secs:.2f} 秒"
    else:
        return f"{secs:.2f} 秒"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OceanBase数据插入脚本')

    parser.add_argument('--lang', nargs='+', default=['en'],
                       help='要处理的语言列表，支持: ar de en es fr hi it ja ko pt ru th zh')
    parser.add_argument('--all', action='store_true',
                       help='处理所有支持的语言')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（YAML 格式），未指定则使用默认配置')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='覆盖配置文件中的批量插入大小')
    
    args = parser.parse_args()

    config = load_oceanbase_insert_config(args.config)

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.all:
        languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    else:
        languages = check_languages(args.lang)

    for lang in languages:
        logging.info(f"开始处理语言: {lang}")
        try:
            test_simple_insert(lang, config)
            logging.info(f"✓ 语言 {lang} 处理完成！")
        except Exception as e:
            logging.error(f"✗ 语言 {lang} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()