#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## 这个作为实际的插入测试，插入200000条数据，然后创建索引

import os
import pymysql
import numpy as np
import argparse
from mldr_common_tools import load_corpus, fvecs_read_yield, check_languages, get_default_query_result_dir

def test_simple_insert(lang: str):
    """插入数据测试"""
    
    # 数据库配置
    DB_CONFIG = {
        'host': '127.0.0.1',
        'port': 2881,
        'user': 'root',
        'password': '',
        'database': 'test',
        'charset': 'utf8mb4'
    }
    
    # 连接数据库
    print("连接数据库...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 创建测试表
    print("创建测试表...")
    table_name = f"test_insert_{lang}"
    
    # 先删除已存在的表
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_sql)
    
    # 创建表SQL - 先不创建索引，等数据插入完成后再创建
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        id BIGINT AUTO_INCREMENT,
        base_id VARCHAR(255) NOT NULL,
        docid_col VARCHAR(255) NOT NULL,
        fulltext_col LONGTEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    cursor.execute(create_table_sql)
    print("测试表创建成功")
    
    # 加载语料库 - 使用流式加载以减少内存占用
    print(f"加载语料库: {lang}")
    corpus = load_corpus(lang)
    
    # 先获取总数，然后分批处理
    try:
        actual_corpus_count = len(corpus)
    except:
        # 如果无法直接获取长度，尝试获取第一条来估算
        actual_corpus_count = corpus.num_rows if hasattr(corpus, 'num_rows') else 200000
    
    # 选择较小的值：实际条数或200000
    test_count = min(200000, actual_corpus_count)
    print(f"将插入 {test_count} 条记录")
    
    # 分批处理以减少内存占用
    batch_size = 1000
    inserted_count = 0
    
    for batch_start in range(0, test_count, batch_size):
        batch_end = min(batch_start + batch_size, test_count)
        batch_indices = list(range(batch_start, batch_end))
        
        # 只加载当前批次的数据
        batch_docids = [corpus[i]["docid"] for i in batch_indices]
        batch_texts = [corpus[i]["text"] for i in batch_indices]
        
        for idx, (docid, text) in enumerate(zip(batch_docids, batch_texts)):
            i = batch_start + idx
            try:
                # 获取数据
                # 生成base_id 
                base_group_size = 500
                base_id_number = (i // base_group_size) + 1
                base_id = f"base_id_{base_id_number}"
                
                insert_sql = f"""
                INSERT INTO {table_name} (base_id, docid_col, fulltext_col)
                VALUES (%s, %s, %s)
                """
                cursor.execute(insert_sql, (base_id, docid, text))
                
                # print(f"成功插入第 {i+1} 条记录")
                inserted_count += 1
                
            except Exception as e:
                print(f"插入第 {i+1} 条记录失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 每批次提交一次，减少内存占用
        conn.commit()
        
        # 显式释放批次数据
        del batch_docids, batch_texts
        
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"已处理 {batch_end}/{test_count} 条记录")
    
    # 显式释放语料库数据
    del corpus
    import gc
    gc.collect()
    
    
    # 最终提交（虽然每批次已提交，但确保所有数据都已提交）
    conn.commit()
    
    print("验证插入结果...")
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f" 表中总记录数: {count}")
    
    # cursor.execute(f"SELECT id, base_id, docid_col, fulltext_col FROM {table_name} LIMIT 5")
    # records = cursor.fetchall()
    # print("前5条记录:")
    # for record in records:
    #     print(f"  ID: {record[0]}, BaseID: {record[1]}, DocID: {record[2]}, Text: {record[3]}...")
    
    print("创建普通索引...")
    try:
        cursor.execute(f"CREATE /*+ parallel(90) */ INDEX idx_base_id ON {table_name}(base_id)")
        print("base_id索引创建成功")
    except Exception as e:
        print(f"base_id索引创建失败: {e}")
    
    try:
        cursor.execute(f"CREATE /*+ parallel(90) */ INDEX idx_docid ON {table_name}(docid_col)")
        print("docid索引创建成功")
    except Exception as e:
        print(f"docid索引创建失败: {e}") 
    
    # 2. 创建全文索引
    print("创建全文索引...")
    try:
        cursor.execute(f"CREATE /*+ parallel(90) */ FULLTEXT INDEX ft_fulltext ON {table_name}(fulltext_col)")
        print("全文索引创建成功")
    except Exception as e:
        print(f"全文索引创建失败: {e}")
    
    print("所有索引创建完成！")
    
    # 清理
    cursor.close()
    conn.close()
    print("测试完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速插入测试脚本')
    parser.add_argument('--lang', nargs='+', default=['en'], 
                       help='要处理的语言列表，支持: ar de en es fr hi it ja ko pt ru th zh')
    parser.add_argument('--all', action='store_true', 
                       help='处理所有支持的语言')
    
    args = parser.parse_args()
    
    # 确定要处理的语言列表
    if args.all:
        languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
        print(f"使用 --all 参数，将处理所有 {len(languages)} 种语言")
    else:
        languages = check_languages(args.lang)
        print(f"将处理指定的 {len(languages)} 种语言: {languages}")
    
    # 逐个处理每种语言
    for lang in languages:
        print(f"\n{'='*60}")
        print(f"开始处理语言: {lang}")
        print(f"{'='*60}")
        
        try:
            test_simple_insert(lang)
            print(f"语言 {lang} 处理完成！")
        except Exception as e:
            print(f"语言 {lang} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"继续处理下一个语言...")
            continue
    
    print(f"\n{'='*60}")
    print("所有语言处理完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 